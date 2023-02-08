import argparse
import pathlib
import numpy as np
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    return parser

def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    
    for idx in range(len(dataset)):
        data_list = []
        data_list.append(dataset[idx])
        batch_labels, batch_strs, batch_tokens = batch_converter(data_list)
        
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

        if (len(batch_tokens[0]) < 1024):
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            out = np.array(results["representations"][33].cpu())
            args.output_file = args.output_dir / f"{batch_labels[0]}.pt"
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            result = {"label": batch_labels[0]}
            result["representations"] = out[:, 1:len(batch_strs[0])+1, :] 
            torch.save(
                result,
                args.output_file,
            )
        else:
            windows = 5
            length = 512
            out = np.zeros((1, len(batch_tokens[0]), 1280))
            weight = np.zeros((1, len(batch_tokens[0]), 1280))
            i = int(np.ceil((len(batch_tokens[0])-length)/windows))
            for s in range(i+1):
                start = s * windows
                end = min(start+length, len(batch_tokens[0]))
                
                temp_seq = batch_tokens[:, start:end]
                with torch.no_grad():
                    results = model(temp_seq, repr_layers=[33], return_contacts=False)
                token_representations = np.array(results["representations"][33].cpu())
                out[:, start:end] += token_representations[:, :]
                weight[:, start:end] += 1
                if end == len(batch_tokens[0]):
                    break
            out /= weight
            args.output_file = args.output_dir / f"{batch_labels[0]}.pt"
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            result = {"label": batch_labels[0]}
            result["representations"] = out[:, 1:len(batch_strs[0])+1, :]
            torch.save(
                result,
                args.output_file,
            )
        
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
