import argparse
import torch
import numpy as np
from copy import deepcopy
import shutil
from pathlib import Path

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_supervised_finetuning import Exp_Supervised_Finetuning
from utils.tools import set_seed, print_formatted_dict
from data_provider.data_loader import update_args_from_dataset


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFCNet")
    
    
    parser.add_argument(
    '--use_full_tpe',
    action='store_true',
    default=True,  # 默认使用新的协同模块
    help='use full TPE core with cross-module attention'
)


    # 确保所有数据集都启用约束
    parser.add_argument(
        "--apply_physical_constraints",
        action="store_true",
        default=True,  # 默认启用
        help="apply physical constraints to all datasets",
)
    
    # 在 get_args_from_parser() 函数中添加以下参数：
    parser.add_argument(
        '--memory_enable',
        action='store_true',
        default=True,
        help='enable temporal pattern memory'
)
    
    parser.add_argument(
    '--ablation',
    type=str,
    default="none",
    choices=["none", "no_temporal_memory", "no_freq_enhance", "no_physical_constraints", "no_two_stage", "no_prompt"],
    help="specify ablation experiment"
)

    parser.add_argument(
    '--no_temporal_memory',
    action='store_true',
    default=False,
    help='disable temporal pattern memory module'
)

    parser.add_argument(
    '--no_freq_enhance',
    action='store_true',
    default=False,
    help='disable frequency enhancement module'
)

    parser.add_argument(
    '--no_physical_constraints',
    action='store_true',
    default=False,
    help='disable physical constraints'
)

    parser.add_argument(
    '--no_two_stage',
    action='store_true',
    default=False,
    help='disable two-stage training (skip supervised finetuning)'
)

    parser.add_argument(
    '--no_prompt',
    action='store_true',
    default=False,
    help='disable prompt prefix'
)
    parser.add_argument(
        '--num_patterns',
        type=int,
        default=10,
        help='number of temporal patterns to remember'
)
    parser.add_argument(
        '--pattern_length',
        type=int,
        default=24,
        help='length of each temporal pattern'
)
    parser.add_argument(
        '--memory_temperature',
        type=float,
        default=0.1,
        help='temperature for memory attention'
)
    
    parser.add_argument(
        '--freq_enable',
        action='store_true',
        default=True,
        help='enable frequency domain enhancement'
)
    parser.add_argument(
        '--freq_ratio',
        type=float,
        default=0.5,
        help='ratio of frequency domain information'
)
    
    # * 新增: 提示前缀配置
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="path to prompt file to prepend to input data"
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=0,
        help="length of prompt tokens to use (0 means use all)"
    )

    # * basic config
    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
        help="time-series task",
        choices=["long_term_forecast"],
    )
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="TFCNet",
        help="model name",
    )
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
    )
    parser.add_argument(
        "--delete_checkpoints",
        action="store_true",
        help="delete checkpoints after training",
        # default=False,
        default=True,
    )

    # * data loader
    parser.add_argument(
        "--data_name",
        type=str,
        default="ETTh1",
        help="dataset name",
        choices=[
            "Weather",
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "ECL",
            "Traffic",
        ],
    )
    parser.add_argument("--data", type=str, default="ETTh1", help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate (only for downstream tasks)",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--pred_len_list",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
        help="the prediction length list",
    )
    parser.add_argument(
        "--percent",
        type=int,
        default=100,
        help="the percentage of the training set",
    )

    # * forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=0, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # * model architecture
    parser.add_argument(
        "--LLM",
        type=str,
        default="gpt2",
        help="the pretrained LLM model",
        choices=["gpt2", "llama"],
    )
    parser.add_argument(
        "--no_freeze",
        action="store_true",
        help="if False, we will freeze the parameters of the pretrained LLM model",
        default=False,
    )
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="if False, we will use the pretrained weights of the LLM model",
        default=False,
    )
    parser.add_argument(
        "--first_k_layers",
        type=int,
        default=6,
        help="the number of initial layers to be used in LLM",
    )
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size (C)")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--token_embed_type",
        type=str,
        default="conv",
        choices=["linear", "conv"],
        help="token embedding type",
    )
    parser.add_argument(
        "--token_embed_kernel_size",
        type=int,
        default=3,
        help="token embedding kernel size (for conv)",
    )
    parser.add_argument(
        "--temporal_embed_type",
        type=str,
        default="learned",
        choices=["none", "fixed", "learned", "timeF"],
        help="temporal embedding type",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--patch_len", type=int, default=16, help="the length of the patch"
    )
    parser.add_argument(
        "--stride", type=int, default=16, help="the stride of the patch"
    )

    # * peft (LoRA-related)
    parser.add_argument(
        "--peft_method",
        type=str,
        default="lora",
        choices=["none", "lora", "adalora"],
        help="PEFT method",
    )
    parser.add_argument(
        "--peft_params_r",
        type=int,
        default=8,
        help="the dimension of the low-rank matrices",
    )
    parser.add_argument(
        "--peft_params_lora_alpha",
        type=int,
        default=32,
        help="the scaling factor for the low-rank matrices",
    )
    parser.add_argument(
        "--peft_params_lora_dropout",
        type=float,
        default=0.1,
        help="the dropout probability of the LoRA layers",
    )

    # * training_stage_params (sft)
    parser.add_argument(
        "--enable_supervised_finetuning",
        type=bool,
        default=True,
        help="enable supervised finetuning (sft)",
    )
    parser.add_argument(
        "--sft_optim",
        type=str,
        default="Adam",
        help="optimizer",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--sft_learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--sft_lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--sft_weight_decay", type=float, default=0.001)
    parser.add_argument("--sft_train_epochs", type=int, default=10, help="train epochs")

    # * training_stage_params (dft)
    parser.add_argument(
        "--dft_optim",
        type=str,
        default="Adam",
        help="optimizer",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--dft_learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--dft_lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--dft_weight_decay", type=float, default=0.001)
    parser.add_argument("--dft_train_epochs", type=int, default=10, help="train epochs")

    # * training_stage_params (shared)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--delta", type=float, default=0.0001, help="early stopping delta"
    )
    parser.add_argument(
        "--ft_mode",
        type=str,
        default="lp_ft",
        choices=["lp_ft", "lp", "ft"],
        help="fine-tuning mode (it should be `ft` for sft and `lp_ft` for dft)",
    )

    # * Hardware
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        # default=False,
        default=True,
    )

    args, _ = parser.parse_known_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.root_path = Path.cwd()  # Set this outside of the trainable function

    args.return_single_feature = (
        True  # make batch_size invariant to the number of features
    )

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    for key, value in tunable_params.items():
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args(args, fixed_params, tunable_params):
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
    args = update_args_from_dataset(args)

# 新增：根据消融实验设置覆盖参数
    if args.ablation != "none":
        print(f"🧪 执行消融实验: {args.ablation}")
        
        if args.ablation == "no_temporal_memory":
            args.memory_enable = False
            print("🚫 移除时间模式记忆模块")
            
        elif args.ablation == "no_freq_enhance":
            args.freq_enable = False
            print("🚫 移除频率增强模块")
            
        elif args.ablation == "no_physical_constraints":
            args.apply_physical_constraints = False
            print("🚫 移除物理约束")
            
        elif args.ablation == "no_two_stage":
            args.enable_supervised_finetuning = False
            print("🚫 移除两阶段训练")
            
        elif args.ablation == "no_prompt":
            args.prompt_file = None
            print("🚫 移除提示前缀")
    
    # 处理单独的消融参数（兼容旧版命令行）
    if args.no_temporal_memory:
        args.memory_enable = False
        print("🚫 移除时间模式记忆模块")
        
    if args.no_freq_enhance:
        args.freq_enable = False
        print("🚫 移除频率增强模块")
        
    if args.no_physical_constraints:
        args.apply_physical_constraints = False
        print("🚫 移除物理约束")
        
    if args.no_two_stage:
        args.enable_supervised_finetuning = False
        print("🚫 移除两阶段训练")
        
    if args.no_prompt:
        args.prompt_file = None
        print("🚫 移除提示前缀")

    args.setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_eb{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.embed,
    )
    print(f"Args in experiment: {args}")

    # Create sft_args
    sft_args = deepcopy(args)
    sft_args.task_name = "supervised_finetuning"
    sft_args.ft_mode = "ft"  # no need to probe
    sft_args.features = "M" # there's no univariate task in sft
    sft_args.pred_len = sft_args.stride
    sft_args.label_len = sft_args.seq_len - sft_args.pred_len
    sft_args.setting = "sft_" + sft_args.setting

    # Create dft_args (just like args)
    dft_args = deepcopy(args)

     # 传递提示前缀参数到sft和dft参数
    sft_args.prompt_file = args.prompt_file
    sft_args.prompt_length = args.prompt_length
    dft_args.prompt_file = args.prompt_file
    dft_args.prompt_length = args.prompt_length

    return sft_args, dft_args


def get_exp(args):
    # Downstream task
    if args.task_name == "long_term_forecast":
        exp = Exp_Long_Term_Forecast(args)
    elif args.task_name == "supervised_finetuning":
        exp = Exp_Supervised_Finetuning(args)
    else:
        raise NotImplementedError

    return exp


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
) -> dict:
    # Update args
    sft_args, dft_args = update_args(args, fixed_params, tunable_params)

    # sft
    if dft_args.enable_supervised_finetuning:
        sft_exp = get_exp(sft_args)
        sft_metrics = sft_exp.train(use_tqdm=True)

    # dft
    dft_metrics_dict = {}


    for pred_len in dft_args.pred_len_list:
        # Update pred_len
        dft_args.pred_len = pred_len

        dft_exp = get_exp(dft_args)
        dft_metrics_dict[pred_len] = dft_exp.train(use_tqdm=True)

    # 新增：收集所有预测结果
    all_predictions = {}
    
    for pred_len in dft_args.pred_len_list:
        # Update pred_len
        dft_args.pred_len = pred_len

        dft_exp = get_exp(dft_args)
        dft_metrics_dict[pred_len] = dft_exp.train(use_tqdm=True)
        
        # 新增：实验完成后立即保存预测值
        print(f"\n📊 保存 {pred_len} 步长的预测结果...")
        
        # 获取测试数据
        _, test_loader = dft_exp._get_data(flag="test")
        
        # 直接获取预测结果
        preds, trues = get_predictions(dft_exp, test_loader, pred_len)
        
        if preds is not None:
            # 保存到字典
            all_predictions[pred_len] = {
                'predictions': preds,
                'ground_truths': trues
            }
            
            # 立即保存到文件
            save_to_file(
                preds, trues, 
                data_name=dft_args.data_name,
                pred_len=pred_len,
                save_dir="./predictions"
            )

    # Return metrics
    return_metrics = {}
    return_metrics["avg_mse"] = np.mean(
        [v["best_test_loss"] for v in dft_metrics_dict.values()]
    )
    return_metrics["avg_mae"] = np.mean(
        [v["best_test_mae"] for v in dft_metrics_dict.values()]
    )
    for pred_len in dft_args.pred_len_list:
        return_metrics[f"{pred_len}_mse"] = dft_metrics_dict[pred_len]["best_test_loss"]
        return_metrics[f"{pred_len}_mae"] = dft_metrics_dict[pred_len]["best_test_mae"]

    if args.delete_checkpoints:
        # Delete both sft and dft checkpoints
        shutil.rmtree(args.checkpoints)

    print("\n✅ 实验完成！预测值已保存到 ./predictions/ 目录")
    
    return return_metrics


def get_predictions(exp, data_loader, pred_len):
    """直接获取预测值和真实值"""
    import numpy as np
    
    all_predictions = []
    all_ground_truths = []
    
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(exp.device)
            )
            
            # forward
            with torch.cuda.amp.autocast(enabled=exp.args.use_amp):
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if exp.args.features == "MS" else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:]

            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(batch_y.cpu().numpy())

    if all_predictions:
        return np.concatenate(all_predictions, axis=0), np.concatenate(all_ground_truths, axis=0)
    return None, None


def save_to_file(predictions, ground_truths, data_name, pred_len, save_dir="./predictions"):
    """保存到文件"""
    import os
    import numpy as np
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{data_name}_pred{pred_len}_{timestamp}.npz"
    filepath = os.path.join(save_dir, filename)
    
    # 保存
    np.savez_compressed(
        filepath,
        predictions=predictions,
        ground_truths=ground_truths
    )
    
    print(f"  保存到: {filepath}")
    print(f"  预测值形状: {predictions.shape}")
    print(f"  真实值形状: {ground_truths.shape}")    


if __name__ == "__main__":
    """------------------------------------"""
    # data_name = "Weather"  # 21
    data_name = "ETTh1"  # 7
    # data_name = "ETTh2"  # 7
    # data_name = "ETTm1"  # 7
    # data_name = "ETTm2"  # 7
    # data_name = "ECL"  # 321
    # data_name = "Traffic"  # 862

    # pred_len_list = [96, 192, 336, 720]
    # pred_len_list = [96, 192, 336]  # 5% (ETTh1, ETTh2, Traffic)
    # pred_len_list = [24, 48, 168, 336, 720] # linear probe
    pred_len_list = [96]
    # pred_len_list = [192]
    # pred_len_list = [336]
    # pred_len_list = [720]

    percent = 100
    # percent = 10
    # percent = 5

    # num_workers = 4
    # num_workers = 6
    num_workers = 8

    batch_size = 128  # 8G
    # batch_size = 512 # 24G
    """------------------------------------"""
    set_seed(seed=2023)

    # Setup args
    args = get_args_from_parser()

    # Setup fixed params
    fixed_params = {
        "data_name": data_name,
        "pred_len_list": pred_len_list,
        "percent": percent,
        "num_workers": num_workers,
        "batch_size": batch_size,
    }

    # Setup tunable params
    # TODO: copy `config` from `exp_settings_and_results` (be careful with the boolean values)
    tunable_params = {
        "enable_supervised_finetuning": True,
        "first_k_layers": 6,
        "patch_len": 16,
        "stride": 8,
        "seq_len": 336,
        "ft_mode": "lp_ft",
        "dropout": 0.05,
        "token_embed_type": "conv",
        "token_embed_kernel_size": 3,
        "temporal_embed_type": "learned",
        "freq": "h",
        # "peft_method": "lora",
        "peft_method": "adalora",
        "sft_optim": "AdamW",
        "sft_learning_rate": 7.912045141879411e-05,
        "sft_lradj": "constant",
        "sft_weight_decay": 0.0005542494992024964,
        "sft_train_epochs": 5,
        "dft_optim": "AdamW",
        "dft_learning_rate": 1.8257759510439175e-05,
        "dft_lradj": "constant",
        "dft_weight_decay": 0.0014555863788252605,
        "dft_train_epochs": 15,
        "peft_params_r": 8,
        "peft_params_lora_alpha": 64,
        "peft_params_lora_dropout": 0,
    }

    # Run
    return_metrics = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(return_metrics)