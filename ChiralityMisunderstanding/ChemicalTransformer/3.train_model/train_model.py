import sys, os
import argparse
import datetime, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import shutil
from pytorch_memlab import MemReporter
import matplotlib.pyplot as plt
from apex import amp

sys.path.append("../../tools")
from path import get_root_path
ROOT_PATH = get_root_path()

sys.path.append(ROOT_PATH+"/miz2022y/github/ChemicalTransformer")
from src.data.buckets import Buckets
from src.models.model_whole import Whole, Whole2
from src.models.components import Generator, Embeddings, FactorizedEmbeddings, \
    NoamOpt, LabelSmoothing, \
    WeightSharingEncoder, WeightSharingDecoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def main():
    print(f"DEVICE: {DEVICE}")
    parser = argparse.ArgumentParser(description="Train Transformer")

    parser.add_argument("--trainbucketfile", type=str, nargs='*', help="File name of pickled bucket or directory name of saved bucket for training.")
    parser.add_argument("--valbucketfile", type=str, help="File name of pickled bucket or directory name of saved bucket for validation.")
    parser.add_argument("--wshare", action="store_true", help="If specified, weight sharing is applied to Transformer (specification in ALBERT).")
    parser.add_argument("--femb",  action="store_true", help="If specified, embedding matrix is factorized (specification in ALBERT).")
    parser.add_argument("--amp", action="store_true", help="If specified use NVIDIA/apex to save memory and time.")
    parser.add_argument("--optimizer", default='adam', help="Specify either 'adam' or 'adamw'")
    parser.add_argument("--tokenizer", type=str, help="Pickled file of tokenizer.")
    parser.add_argument("--splitemb", action="store_true", help="If specified, embedding layer is not shared for source and target.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_tokens", type=int)
    parser.add_argument("--niter", type=int)
    parser.add_argument("--abortvalperfect",default=1, type=float, help="Abort learing when perfect accuracy of validation set exceeded the specified value.")
    parser.add_argument("--dmodel", type=int, default=512)
    parser.add_argument("--demb", type=int, default=128)
    parser.add_argument("--dff", type=int, default=2048)
    parser.add_argument("--modelfile", type=str, help="model is initialized by specified file(assumed to be pickle)")
    parser.add_argument("--optfile", type=str, help="model_opt is initialized by specified file (assumed to be pickle)")
    parser.add_argument("--memstep", type=int)
    parser.add_argument("--evalstep", type=int)
    parser.add_argument("--studyname", type=str, default="default_study")
    parser.add_argument("--optstep", type=int, default=2)
    parser.add_argument("--recordstep", type=int)
    args = parser.parse_args()

    #make directory to save results.
    dt_now = datetime.datetime.now()
    datestring = f"{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}"
    result_dirname = f"training_results/{datestring}_{args.studyname}/"
    if os.path.exists(result_dirname):
        shutil.rmtree(result_dirname)
    os.makedirs(result_dirname, exist_ok=True)
    os.makedirs(result_dirname+"models/", exist_ok=True)
    os.makedirs(result_dirname+"model_opts/", exist_ok=True)
    
    # Parameter setting
    n_tokens = args.n_tokens if args.n_tokens is not None else 12500
    n_batch_iter = args.niter if args.niter is not None else 100000
    perfect_val_abort = args.abortvalperfect
    d_model = args.dmodel
    emb_dim = args.demb
    dim_feedforward = args.dff
    log_step = 1000
    memory_step =args.memstep if args.memstep is not None else 1000
    val_step = args.evalstep if args.evalstep is not None else 2000
    record_step = args.recordstep if args.recordstep is not None else 5000
    opt_step = args.optstep
    nhead=8
    layer_norm_eps = 1e-5
    dropout = 0.1
    activation = "relu"
    tokenizer_path = args.tokenzier if args.tokenizer is not None else \
        "../1.make_tokenizer/smiles_tokenizer.pkl"
    fix_seed(args.seed)

    train_bucket_paths = args.trainbucketfile if args.trainbucketfile is not None else \
        ["../2.make_dataset/train"]
    val_bucket_path = args.valbucketfile if args.valbucketfile is not None else \
        "../2.make_dataset/val"
    if val_bucket_path[len(val_bucket_path)-4:] == ".pkl":
        with open(val_bucket_path, mode='rb') as f:
            buckets_val = pickle.load(f)
    else:
        buckets_val = Buckets.load(val_bucket_path)
    n_val = len(buckets_val)
    with open(tokenizer_path, mode='rb') as f:
        tokenizer = pickle.load(f)
    n_tok = tokenizer.n_tok()

    #Save parameters
    param_names = ['trainbucketfile', 'valbucketfile', 'modelfile', 'optfile','tokenizer_path', 'seed', 'test', 'n_tokens', 'n_batch_iter','perfect_val_abort',
        'splitemb', 'd_model', 'emb_dim', 'dim_feedforward', 'wshare', 'femb', 
        'log_step', 'memory_step', 'val_step', 'record_step', 'opt_step',
        'nhead', 'layer_norm_eps', 'dropout', 'activation', 'optimizer']
    params = []
    for param_name in param_names:
        try:
            params.append(str(eval(param_name)))
        except NameError:
            params.append(str(eval('args.'+param_name)))
    params = pd.Series(data=params, index=param_names, name='value')
    params.to_csv(result_dirname+"parameters.csv")

    # make model
    if args.modelfile is None:
        if args.wshare:
            encoder_layer = nn.modules.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                            activation=activation)
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            encoder = WeightSharingEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
            decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                            activation)
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            decoder = WeightSharingDecoder(decoder_layer, num_layers=6, norm=decoder_norm)
            transformer = nn.Transformer(d_model, nhead, custom_encoder=encoder, custom_decoder=decoder)
        else:
            transformer = nn.Transformer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead)
        generator = Generator(d_model, n_tok)
        if args.splitemb:
            if args.femb:
                emb_src = FactorizedEmbeddings(d_model, n_tok, emb_dim=emb_dim)
                emb_tgt = FactorizedEmbeddings(d_model, n_tok, emb_dim=emb_dim)
            else:
                emb_src = Embeddings(d_model, n_tok)
                emb_tgt = Embeddings(d_model, n_tok)
            model = Whole2(transformer, None, generator, d_model, DEVICE, emb_src, emb_tgt, )
        else:
            if args.femb:
                emb = FactorizedEmbeddings(d_model, n_tok, emb_dim=emb_dim)
            else:
                emb = Embeddings(d_model, n_tok)
            model = Whole(transformer, emb, generator, d_model, device=DEVICE)
        for layer in [emb, generator]:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    else:
        with open(args.modelfile, mode='rb') as f:
            model = pickle.load(f)
    model.to(DEVICE)
    criterion = LabelSmoothing(size=n_tok, padding_idx=0, smoothing=0.1)
    if args.optfile is None:
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        else:
            raise ValueError(f"Unsupported argument of --optimizer: {args.optimizer}")
        model_opt = NoamOpt(d_model, warmup=4000, optimizer=optimizer)
        if args.amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    else:
        with open(args.optfile, mode='rb') as f:
            model_opt = pickle.load(f)

    # training
    len_hist = []
    loss_hist = []
    perfect_hist = []
    partial_hist = []
    start = time.time()
    total_loss = 0
    total_tokens = 0
    model.train()
    optimizer.zero_grad()
    perfect_val = 0

    with open(result_dirname+f"models/0.pkl", mode='wb') as f:
        pickle.dump(model, f)
    print("Training started.")
    epoch = 0
    step_now = 0
    while True:
        if epoch == 0 or len(train_bucket_paths) > 1:
            print("Loading buckets...")
            bucket_path = train_bucket_paths[epoch%len(train_bucket_paths)]
            if bucket_path[len(bucket_path)-4:] == '.pkl':
                with open(bucket_path, mode='rb') as f:
                    buckets_train = pickle.load(f)
            else:
                buckets_train = Buckets.load(bucket_path)
        for step, (batch_src, batch_tgt) in enumerate(buckets_train.iterbatches(), step_now):
            batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
            batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)
            tgt_n_tokens = (batch_tgt[:,1:] != tokenizer.pad_token).data.sum()
            pred = model.generator(model(batch_src, batch_tgt[:,:-1], pad_token=tokenizer.pad_token))

            loss = criterion(pred.contiguous().view(-1, pred.size(-1)), batch_tgt[:,1:].contiguous().view(-1)) / tgt_n_tokens
            
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            loss = loss.detach().cpu().numpy()
            loss_hist.append(loss)
            len_hist.append(batch_tgt.shape[-1])
            loss *= tgt_n_tokens.detach().cpu().numpy()
            total_loss += loss
            total_tokens += tgt_n_tokens
            del batch_src, batch_tgt, loss
            torch.cuda.empty_cache()
            if (step+1) % opt_step != 0:
                continue

            model_opt.step()
            optimizer.zero_grad()

            if model_opt._step % log_step == 0:
                elapsed = time.time()-start
                loss_per_token = total_loss / total_tokens
                print(f"Step {model_opt._step:5} Loss: {loss_per_token:2.2f} Tokens per Sec: {elapsed/total_tokens}")
                start = time.time()
                total_loss = 0
                total_tokens = 0
            
            if model_opt._step % memory_step == 0:
                print(f"Step {model_opt._step:5} Reserved memory: {torch.cuda.memory_reserved(device=DEVICE):,}") 

            if model_opt._step % val_step == 0:
                print("Validating...")
                model.eval()
                with open(result_dirname+f"models/{model_opt._step}.pkl", mode='wb') as f:
                    pickle.dump(model, f)
                with open(result_dirname+f"model_opts/{model_opt._step}.pkl", mode='wb') as f:
                    pickle.dump(model_opt, f)

                perfect = 0
                partial = 0
                for batch_src, batch_tgt in tqdm(buckets_val.iterbatches()):
                    batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
                    batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)

                    pred = model.greedy_decode(*model.encode(batch_src, pad_token=tokenizer.pad_token),
                        batch_tgt.size(1), start_token=tokenizer.start_token, pad_token=tokenizer.pad_token)
                    pad_mask = (batch_tgt != tokenizer.pad_token).to(torch.int)
                    perfect += torch.all((batch_tgt*pad_mask) == (pred*pad_mask), dim=1).sum().detach().cpu().numpy()
                    partial += (torch.sum((batch_tgt == pred)*pad_mask, dim=1) / torch.sum(pad_mask, dim=1)).sum().detach().cpu().numpy()
                    del batch_src, batch_tgt, pred, pad_mask
                    torch.cuda.empty_cache()
                perfect /= n_val
                partial /= n_val
                print(f"Step {model_opt._step:5} validation Perfect: {perfect:.3f}, Partial: {partial:.3f}")
                perfect_hist.append(perfect)
                partial_hist.append(partial)
                perfect_val = perfect
                model.train()

            if model_opt._step % record_step == 0 or model_opt._step == n_batch_iter or perfect_val >= perfect_val_abort:
                pd.DataFrame(data={"len":len_hist, "loss":loss_hist}).to_csv(result_dirname+"len_loss.csv")
                if len(perfect_hist) > 0:
                    pd.DataFrame(data=perfect_hist, index=(np.arange(len(perfect_hist))+1)*val_step).to_csv(result_dirname+"perfect_val.csv")
                    pd.DataFrame(data=partial_hist, index=(np.arange(len(partial_hist))+1)*val_step).to_csv(result_dirname+"partial_val.csv")
                
                
            if model_opt._step == n_batch_iter or perfect_val >= perfect_val_abort:
                break
        epoch += 1
        step_now=step+1
        if model_opt._step == n_batch_iter or perfect_val >= perfect_val_abort:
            break
        

if __name__ == '__main__':
    main()
