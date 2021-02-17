import math

import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

figure_counter = 0
data_x = []
data_y = []
data_a = []
data_b = []

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_
    self.output_scale = nn.Linear(1, 1)
    nn.init.uniform_(self.output_scale.weight, 1.0, 1.0)
    nn.init.uniform_(self.output_scale.bias, 0.0, 0.0)
    self.output_scale_outcome = nn.Linear(1, 1)
    nn.init.uniform_(self.output_scale_outcome.weight, 1.0, 1.0)
    nn.init.uniform_(self.output_scale_outcome.bias, 0.0, 0.0)

    self._initialize_feature_weights()
    self._zero_virtual_feature_weights()
    self._initialize_affine(self.l1)
    self._initialize_affine(self.l2)
    self._initialize_output()

  def _initialize_feature_weights(self):
    std = 0.1 / math.sqrt(30)
    nn.init.normal_(self.input.weight, 0.0, std)
    nn.init.uniform_(self.input.bias, 0.5, 0.5)

  def _initialize_affine(self, layer):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    std = 0.5 / math.sqrt(fan_in)
    nn.init.normal_(layer.weight, 0.0, std)
    bias = 0.5 - 0.5 * torch.sum(layer.weight, 1)
    layer.bias = nn.Parameter(bias)

  def _initialize_output(self):
    nn.init.uniform_(self.output.bias, 0.0, 0.0)
    std = 1.0 / math.sqrt(L3)
    nn.init.normal_(self.output.weight, 0.0, std)

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    for a, b in self.feature_set.get_virtual_feature_ranges():
      weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def on_after_backward(self):
    w = self.input.weight
    g = w.grad
    a = self.feature_set.features[0].get_factor_base_feature('HalfK')
    b = self.feature_set.features[0].get_factor_base_feature('P')
    g[:, a:b] /= 30.0

  def step_(self, batch, batch_idx, loss_type):
    lim = 1.98
    with torch.no_grad():
      self.l1.weight.clamp_(-lim, lim)
      self.l2.weight.clamp_(-lim, lim)
    us, them, white, black, outcome, score = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    scaling = 250

    q = self(us, them, white, black) * nnue2score / scaling
    # q_score = self.output_scale(q)#.detach())
    q_score = q
    q_outcome = self.output_scale_outcome(q)
    # q_outcome = q
    t = outcome
    # magic = 1.3624 / 0.97558
    magic = 1
    score.clamp_(-3000, 3000)
    # score2 = score + score.sign() * 200.
    p = (score / scaling * magic).sigmoid()

    print('{:.4f} {:.4f} {:.4f} {:.4f}'.format(
        self.output_scale.weight.item(), self.output_scale.bias.item(),
        self.output_scale_outcome.weight.item(), self.output_scale_outcome.bias.item(),
    ))

    global figure_counter
    figure_counter += 1
    if figure_counter == 100:
      figure_counter = 0
      data_x.extend((q_score * scaling).detach().cpu().numpy())
      data_y.extend(score.detach().cpu().numpy())
      data_a.extend((q_outcome * scaling).detach().cpu().numpy())
      data_b.extend(outcome.detach().cpu().numpy())

      if len(data_x) > 100000:
        print('Save fig')
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.scatter(data_x, data_y, s=0.1)
        plt.subplot(1, 2, 2)
        plt.scatter(data_a, data_b, s=0.1)
        plt.savefig('runs/run22/plot.jpg')
        plt.close()
        data_x.clear()
        data_y.clear()
        data_a.clear()
        data_b.clear()

    with torch.no_grad():
      lambdas = torch.where(torch.logical_or(score <= -1000, score >= 1000), 0.2, self.lambda_)
      # lambdas = self.lambda_
      # multiplier = torch.where(score.sign() == q_score.sign(), 0.5, 1.0)
      # multiplier *= torch.where(q_score.abs() < score.abs(), 1.0, 0.5)
      # multiplier *= torch.where(torch.logical_and(
      #     score.sign() == q_score.sign(),
      #     torch.logical_and(score.abs() >= 1000, q_score.abs() >= 1000)
      # ), 0.0, 1.0)

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q_score) + (1.0 - p) * F.logsigmoid(-q_score))
    outcome_loss = -(t * F.logsigmoid(q_outcome) + (1.0 - t) * F.logsigmoid(-q_outcome))
    # teacher_loss = F.mse_loss(q_score.sigmoid(), p)
    # outcome_loss = F.mse_loss(q_outcome.sigmoid(), t)

    # teacher_loss *= multiplier
    # teacher_entropy *= multiplier

    result  = lambdas * teacher_loss    + (1.0 - lambdas) * outcome_loss
    entropy = lambdas * teacher_entropy + (1.0 - lambdas) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)

    # mask = torch.logical_and(-1000 < score, score < 1000)
    # mse_loss = torch.masked_select(F.mse_loss(q_score * scaling, score, reduction='none'), mask).mean()
    mse_loss = F.mse_loss(torch.clamp_(q_score * scaling, -1000, 1000), torch.clamp_(score, -1000, 1000))
    self.log('mse_loss', mse_loss)

    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    # Train with a lower LR on the output layer
    LR = 5e-0
    train_params = [
      # {'params': self.get_layers(lambda x: self.input == x), 'lr': LR},
      # {'params': self.get_layers(lambda x: self.l1 == x), 'lr': LR / 2},
      # {'params': self.get_layers(lambda x: self.input != x and self.l1 != x), 'lr': LR / 10},
      # {'params': self.get_layers(lambda x: self.input != x), 'lr': LR / 10},
      # {'params': self.get_layers(lambda x: self.input != x), 'lr': LR / 10, 'momentum': 0.9, 'nesterov': True},
      {'params': self.get_layers(lambda x: self.output != x), 'lr': LR},
      {'params': self.get_layers(lambda x: self.output == x), 'lr': LR / 10},
      # {'params': self.get_layers(lambda x: True), 'lr': LR},
    ]
    # increasing the eps leads to less saturated nets with a few dead neurons
    # optimizer = ranger.Ranger(train_params, betas=(.9, 0.999), eps=1.0e-7)
    optimizer = torch.optim.SGD(train_params, lr=LR)

    def get_lr(epoch):
        # if epoch == 0:
        #     return 0.1
        # if epoch == 1:
        #     return 0.2

        # return 0.1 * (epoch + 1)

        # if epoch == 0:
        #     return 0.1

        # return 1.0 / (10.0 ** (epoch // 1))

        # if epoch % 10 == 8:
        #     return 0.5
        # if epoch % 10 == 9:
        #     return 0.25

        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    # Drop learning rate after 75 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.3)
    return [optimizer], [scheduler]

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
