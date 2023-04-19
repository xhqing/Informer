from Informer.utils.tools import dotdict
import torch

args = dotdict()

args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.root_path = './' # root path of data file
args.data_path = 'ETTh1.csv' # data file

args.scale = True

args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT' # target feature in S or MS task
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h

args.seq_len = 30*24           # input sequence length of Informer encoder
args.label_len = 7*24      # start token length of Informer decoder
args.pred_len = 24                # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 7   # encoder input size
args.dec_in = 7   # decoder input size
args.c_out = 1          # output size

args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 2     # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0. # dropout
args.attn = 'full' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0

args.batch_size = 32
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1' # learning rate adjust
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.train_epochs = 6
args.patience = 3

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0 # device string: cuda:0

args.use_multi_gpu = False
args.devices = '0,1,2,3'
