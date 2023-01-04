import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--outfolder", type=str, default='out', help="name of output folder")
    parser.add_argument("--dataset_name", type=str, default='FLIR', help="name of dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="path to dataset")
    parser.add_argument("--prediction_path", type=str, default=None, help="path to model predictions")
    parser.add_argument("--fusion_method", type=str, default="middle_fusion", 
                    choices=['rgb_only','thermal_only', 'early_fusion', 'middle_fusion'], help="Which fusion method to use?")
    parser.add_argument("--model_path", type=str, default=None, help="path to trained model")
    parser.add_argument("--score_fusion", type=str, default="probEn",
                    choices=['avg','max', 'probEn'], help="Which fusion method to use?")
    parser.add_argument("--box_fusion", type=str, default="v-avg", 
                    choices=['avg','s-avg', 'v-avg', 'argmax'], help="Which fusion method to use?")
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()