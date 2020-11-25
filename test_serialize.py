import argparse
import halfkp
import model as M
import nnue_dataset
import torch
import pytorch_lightning as pl
import serialize

def test_coalesce(model, dataset):
  batch_size = 1
  stream_factor = nnue_dataset.SparseBatchDataset(halfkp.FACTOR_NAME, dataset, batch_size)
  stream_factor_iter = iter(stream_factor)
  tensors_factor  = next(stream_factor_iter)[:4]
  print(model(*tensors_factor))

  weights = serialize.coalesce_weights(model.input.weight.data)
  model.input.weight = torch.nn.Parameter(weights)

  stream = nnue_dataset.SparseBatchDataset(halfkp.NAME, dataset, batch_size)
  stream_iter = iter(stream_factor)
  tensors = next(stream_iter)[:4]
  print(model(*tensors))

def main():
  parser = argparse.ArgumentParser(description="Tests model conversions.")
  parser.add_argument("model", help="(can be .ckpt, .pt or .nnue)")
  parser.add_argument("dataset")
  args = parser.parse_args()

  def load():
    nnue = M.NNUE.load_from_checkpoint(args.model)
    nnue.eval()
    return nnue

  test_coalesce(load(), args.dataset)

if __name__ == '__main__':
  main()
