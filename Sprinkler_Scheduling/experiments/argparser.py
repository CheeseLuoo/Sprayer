import configargparse
import configparser
import yaml
import sys

def parse_arguments(verbose=False):
    parser = configargparse.ArgParser()
    parser.add_argument("--config",
                        required=False,
                        is_config_file=True,
                        default='./Sprinkler_Scheduling/configs/CONF.yaml',
                        help="Configuration file path.")
    
    # Experiment settings
    parser.add_argument("--strategy_name",
                        type=str,
                        # required=True,
                        default="SA_time_non_uniform",
                        help="Sampling strategy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--verbose",
                        type=bool,
                        default=False,
                        help="Printing details.")
    parser.add_argument("--diffusivity_K",
                        type=float,
                        default=1.2,
                        help="diffusivity coefficient")
    parser.add_argument("--grid_x",
                        type=int,
                        default=20,
                        help="x_range")
    parser.add_argument("--grid_y",
                        type=int,
                        default=20,
                        help="y_range")
    parser.add_argument("--randomsource",
                        type=bool,
                        default=True,
                        help="whether randomsource")
    parser.add_argument("--sourcenum",
                        type=int,
                        default=3,
                        help="number of random pollution source")
    parser.add_argument("--R_change_interval",
                        type=int,
                        default=6,
                        help="time interval of random pollution source change")
    parser.add_argument("--time_co",
                        type=float,
                        default=0.1,
                        help="time step of gaussian process")
    parser.add_argument("--delta_t",
                        type=float,
                        default=3.0,
                        help="interval of environment change")
    parser.add_argument("--sensing_rate",
                        type=float,
                        default=1.0,
                        help="sensing_rate")
    parser.add_argument("--noise_scale",
                        type=float,
                        default=1.0,
                        help="noise_scale")
    parser.add_argument("--num_init_samples",
                        type=int,
                        default=1,
                        help="range of init samples")
    parser.add_argument("--max_num_samples",
                        type=int,
                        default=48,
                        help="total schedule period")
    parser.add_argument("--sche_step",
                        type=int,
                        default=18,
                        help="max schedule steps")
    parser.add_argument("--adaptive_step",
                        type=int,
                        default=3,
                        help="adaptive steps")
    parser.add_argument("--Env",
                        type=bool,
                        default=True,
                        help="dynamic environment.")
    parser.add_argument("--effect_threshold",
                        type=float,
                        default=0.0,
                        help="naive effect_threshold")
    parser.add_argument("--bound1",
                        type=int,
                        default=200,
                        help="simulated annealing round for initual search")
    parser.add_argument("--bound2",
                        type=int,
                        default=20,
                        help="simulated annealing round for optimal search")
    parser.add_argument("--bound3",
                        type=int,
                        default=100,
                        help="simulated annealing round for later search")    
    parser.add_argument("--time_before_sche",
                        type=int,
                        default=5,
                        help="time before sche")
    parser.add_argument("--amplitude",
                        type=float,
                        default=1.0,
                        help="kernal paremeter amplitude")
    parser.add_argument("--lengthscale",
                        type=float,
                        default=0.5,
                        help="kernal paremeter lengthscale")
    parser.add_argument("--init_noise",
                        type=float,
                        default=1.0,
                        help="kernal paremeter init_noise")
    parser.add_argument("--lr_hyper",
                        type=float,
                        default=0.01,
                        help="kernal paremeter lr_hyper")
    parser.add_argument("--lr_nn",
                        type=float,
                        default=0.001,
                        help="kernal paremeter lr_nn")
    parser.add_argument("--team_size",
                        type=int,
                        default=4,
                        help="vehicle size")
    parser.add_argument("--replenish_speed",
                        type=int,
                        default=1,
                        help="replenish speed per time step")
    parser.add_argument("--water_volume",
                        type=int,
                        default=4,
                        help="water volume of one vehicle")
    parser.add_argument('--alpha', type=list, default=[0.75,0.9,1.01,1.05,1.5], required=False, help='object weight.')
    parser.add_argument("--root_dir",
                        type=str,
                        default="./output/",
                        help="Directory for logs.")
    args = parser.parse_args()

    if args.verbose:
        print(parser.format_values())
        
    if sys.platform == 'linux':
        print("当前系统是Linux")
    elif sys.platform == 'win32':
        print("当前系统是Windows")
    
    args.save_name = f'SEED_{args.seed}_X{args.grid_x}_Y{args.grid_y}_VS{args.team_size}_TS{args.max_num_samples}_SS{args.sche_step}_AS{args.adaptive_step}_SN{args.sourcenum}_RS{args.replenish_speed}_WV{args.water_volume}'
        
    return args
