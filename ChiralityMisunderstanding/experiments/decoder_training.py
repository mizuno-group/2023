TOOLS_DIR = "/home/docker/data/miz2022y/tools"
CT_DIR = "/home/docker/data/miz2022y/github/ChemicalTransformer"
RESULT_DIR = "/home/docker/data/miz2022y/NMT_Transformer/zinc/training_results"
DEFAULT_BUCKET_PATH = "/home/docker/data/miz2022y/NMT_Transformer/zinc/220815_buckets"
DEFAULT_TOKENIZER_PATH = "/home/docker/data/miz2022y/NMT_Transformer/zinc/220829_tokenizer"

import sys, os
import argparse
import pickle
import datetime
import time
import numpy as np
import pandas as pd
import torch
sys.path.append(TOOLS_DIR)
sys.path.append(CT_DIR)
from src.models.model_whole import prepare_model, add_required_args
from src.models.components import NoamOpt, LabelSmoothing
from src.data.buckets import Buckets
from notice import notice, noticeerror
from args import save_args

def find_file_s(file_s, prefix):
    if type(file_s) == str:
        return find_file_s([file_s], prefix)[0]
    founds = []
    for file in file_s:
        if os.path.exists(file):
            founds.append(file)
        elif os.path.exists(prefix+file):
            founds.append(prefix+file)
        elif os.path.exists(prefix+"/"+file):
            founds.append(prefix+"/"+file)
        else:
            raise FileNotFoundError(f'Neither "{file}", "{prefix}{file}" nor "{prefix}/{file}" was found.')
    return founds

def load_data(file):
    print(f"Loading {file} ...", end='', flush=True)
    ext = file.split(".")[-1]
    if ext == 'pkl':
        with open(file, mode='rb') as f:
            data = pickle.load(f)
    else:
        data = Buckets.load(file)
    print("loaded.", flush=True)
    return data
    
class data_iterator:
    def __init__(self, files):
        self.epoch = 0
        self.n_file = len(files)
        if self.n_file == 1:
            self.file = load_data(files[0])
        else:
            self.file = None
            self.files = files
    def __iter__(self):
        return self
    def __next__(self):
        self.epoch += 1
        if self.n_file == 1:
            return self.file
        else:
            del self.file
            self.file = load_data(self.files[self.epoch%self.n_files])
            return self.file

def load_tokenizer(args):
    with open(find_file_s(args.tokenizer_file, DEFAULT_TOKENIZER_PATH), mode='rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def load_model(args, DEVICE):
    if args.pretrained_model_file is None:
        model = prepare_model(args, DEVICE)
    else:
        ext = args.pretrained_model_file.split('.')[-1]
        if ext == 'pkl':
            with open(args.pretrained_model_file, mode='rb') as f:
                model = pickle.load(f)
            model.to(DEVICE)
        elif ext == 'pt':
            model = torch.load(args.pretrained_model_file)
            model.to(DEVICE)
        elif ext == 'pth':
            model = prepare_model(args, DEVICE)
            model.load_state_dict(torch.load(args.pretrained_model_file))
        else:
            raise ValueError(f"modelfile with unknown extend: {args.modelfile}")
    return model

def prepare_optimizer(args, parameters):


    if args.optimizer.lower() in ["warmup_adam", "warmup_adamw"]:
        if args.warmup is None:
            args.warmup = 4000
        if args.optimizer.lower() == "warmup_adam":
            base_opt = torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
        else:
            base_opt = torch.optim.AdamW(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
        optimizer = NoamOpt(args.d_model, warmup=args.warmup, optimizer=base_opt)
        optimizer._step = args.optimizer_start_step
    elif args.optimizer.lower() == "adam":
        if args.lr is None:
            args.lr = 0.01
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer.lower() == "adamw":
        if args.lr is None:
            args.lr = 0.01
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(f"Unknown type of args.optimizer: {args.optimizer}")
    return optimizer

@noticeerror
def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {DEVICE}")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--studyname", default="default_study")
    parser.add_argument("--pretrained_model_file")
    parser.add_argument("--tokenizer_file", default='normal_tokenizer.pkl')
    parser.add_argument("--train_file", nargs='+', default=["normal/train.pkl"])
    parser.add_argument("--val_file", nargs='*', default=['normal/val.pkl'])
    add_required_args(parser)
    parser.add_argument("--optimizer", default="warmup_adam")
    parser.add_argument("--lr", default=None)
    parser.add_argument("--optimizer_start_step", default=0)
    parser.add_argument("--warmup", default=None)
    parser.add_argument("--minnstep", type=int, default=0)
    parser.add_argument("--maxnstep", type=int, default=200000)
    parser.add_argument("--minnepoch", type=int, default=0)
    parser.add_argument("--maxnepoch", type=int, default=np.inf)
    parser.add_argument("--minvalperfect",default=1, type=float, help="Abort learing when perfect accuracy of validation set exceeded the specified value.")
    parser.add_argument("--val_file_judge_abort", type=str)
    parser.add_argument("--valstep", type=int, default=2000)
    parser.add_argument("--noticestep", type=int, default=30000)
    parser.add_argument("--savestep", type=int, default=2000)
    parser.add_argument("--logstep", type=int, default=1000)
    parser.add_argument("--keepseed", action="store_true")
    parser.add_argument("--saveallopt", action="store_true")
    parser.add_argument("--optstep", type=int, default=2)
    args = parser.parse_args()

    # make directory
    dt_now = datetime.datetime.now()
    result_dirname_base = RESULT_DIR+f"/{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}_{args.studyname}"
    result_dirname = result_dirname_base
    n_exist = 0
    while os.path.exists(result_dirname):
        print(f"{result_dirname} already exists. Study name was changed to ",
            end='', file=sys.stderr)
        n_exist += 1
        result_dirname = result_dirname_base+str(n_exist)
        print(result_dirname, file=sys.stderr)
    result_dirname += '/'
    os.makedirs(result_dirname, exist_ok=True)
    os.makedirs(result_dirname+"models/", exist_ok=True)

    # prepare tokenizer, model, optimizer, data
    tokenizer = load_tokenizer(args)
    args.n_tok = tokenizer.n_tok()
    train_data_iterator = data_iterator(find_file_s(args.train_file, DEFAULT_BUCKET_PATH))
    if type(args.val_file) == str:
        args.val_file = [args.val_file]
    val_datas = [load_data(val_file) for val_file in find_file_s(args.val_file, DEFAULT_BUCKET_PATH)]
    n_val_datas = len(val_datas)
    if args.val_file_judge_abort is None and n_val_datas > 0:
        args.val_file_judge_abort = args.val_file[0]
        assert args.val_file_judge_abort in args.val_file
    model = load_model(args, DEVICE)
    criterion = LabelSmoothing(size=args.n_tok, padding_idx=tokenizer.pad_token, smoothing=0.1)
    parameters = list(model.transformer.decoder.parameters())+\
        list(model.generator.parameters())
    optimizer = prepare_optimizer(args, parameters)
    save_args(args, result_dirname+"args.csv")

    # training
    loss_steps = []
    loss_hist = []
    val_steps = [0]
    perfect_hist =  {val_file: [0.] for val_file in args.val_file}
    partial_hist = {val_file: [0.] for val_file in args.val_file}
    log_total_tokens = 0
    log_total_loss = 0
    batch_step = 0
    optimizer_step = 0
    model.train()
    optimizer.zero_grad()
    print("training started.")
    try:
        for train_data in train_data_iterator:
            start = time.time()
            for batch_src, batch_tgt in train_data.iterbatches():
                batch_step += 1
                batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
                batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)
                tgt_n_tokens = (batch_tgt[:,1:] != tokenizer.pad_token).data.sum()
                pred = model.generator(model(batch_src, batch_tgt[:,:-1], pad_token=tokenizer.pad_token))

                loss = criterion(pred.contiguous().view(-1, pred.size(-1)), batch_tgt[:,1:].contiguous().view(-1)) / tgt_n_tokens
                loss.backward()
                loss = loss.detach().cpu().numpy()
                loss_hist.append(loss)
                loss_steps.append(optimizer_step)
                loss *= tgt_n_tokens.detach().cpu().numpy()
                log_total_loss += loss
                log_total_tokens += tgt_n_tokens
                del batch_src, batch_tgt, loss
                torch.cuda.empty_cache()
                if (batch_step+1) % args.optstep != 0:
                    continue
                optimizer_step += 1

                optimizer.step()
                optimizer.zero_grad()
                val_start = time.time()

                if optimizer_step % args.logstep == 0:
                    elapsed = time.time()-start
                    loss_per_token = log_total_loss / log_total_tokens
                    print(f"Step {optimizer_step:5} Loss: {loss_per_token:2.2f} Elapsed time: {elapsed:>4.1f}s")
                    start = time.time()
                    log_total_loss = 0
                    log_total_tokens = 0

                if optimizer_step % args.valstep == 0:
                    if n_val_datas == 0:
                        pass
                    else:
                        model.eval()
                        for val_file, val_data in zip(args.val_file, val_datas):
                            if n_val_datas == 1:
                                print("Validating...")
                            else:
                                print(f"Validating {val_file}...")
                            perfect = 0
                            partial = 0
                            for batch_src, batch_tgt in val_data.iterbatches():
                                batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
                                batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)

                                pred = model.greedy_decode(*model.encode(batch_src, pad_token=tokenizer.pad_token),
                                    batch_tgt.size(1), start_token=tokenizer.start_token, pad_token=tokenizer.pad_token)
                                pad_mask = (batch_tgt != tokenizer.pad_token).to(torch.int)
                                perfect += torch.all((batch_tgt*pad_mask) == (pred*pad_mask), dim=1).sum().detach().cpu().numpy()
                                partial += (torch.sum((batch_tgt == pred)*pad_mask, dim=1) / torch.sum(pad_mask, dim=1)).sum().detach().cpu().numpy()
                                del batch_src, batch_tgt, pred, pad_mask
                                torch.cuda.empty_cache()
                            perfect /= len(val_data)
                            partial /= len(val_data)
                            print(f"Step {optimizer_step:5} {'Validation' if n_val_datas == 1 else val_file} Perfect: {perfect:.3f}, Partial: {partial:.3f}")
                            perfect_hist[val_file].append(perfect)
                            partial_hist[val_file].append(partial)
                        val_steps.append(optimizer_step)
                        pd.DataFrame(data=perfect_hist, index=val_steps).to_csv(result_dirname+"perfect_val.csv")
                        pd.DataFrame(data=partial_hist, index=val_steps).to_csv(result_dirname+"partial_val.csv")
                        model.train()

                if optimizer_step % args.noticestep == 0:
                    notice(f"{args.studyname} {optimizer_step}step終了!")
                if optimizer_step % args.savestep == 0:
                    torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
                if (train_data_iterator.epoch < args.minnepoch) or (optimizer_step < args.minnstep) or perfect_hist[args.val_file_judge_abort][-1] < args.minvalperfect:
                    abort_study = False
                else:
                    abort_study = True
                if optimizer_step >= args.maxnstep:
                    abort_study = True
                if abort_study:
                    break
                val_end = time.time()
                start += val_end - val_start

            if abort_study or train_data_iterator.epoch >= args.maxnepoch:
                torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
                pd.DataFrame(data={"loss":loss_hist}, index=loss_steps).to_csv(result_dirname+"loss.csv")
                break
        notice(f'"{args.studyname}"の学習が終わったよ!')
    except KeyboardInterrupt:
        torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
        pd.DataFrame(data={"loss":loss_hist}, index=loss_steps).to_csv(result_dirname+"loss.csv")
        raise KeyboardInterrupt

if __name__ == '__main__':
    main()


