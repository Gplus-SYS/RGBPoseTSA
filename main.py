from utils import parser
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def main():

    args = parser.get_args()
    parser.setup(args)    
    # print(args)
    multi_modality = args.model.get('multi_modality', False)
    double_path = args.model.get('updata_by_which_path', False)

    if args.cls_only:
        if multi_modality:
            from tools.cls import runner_cls_MM as runner
        else:
            from tools.cls import runner_cls as runner
    elif args.ps:
        from tools.ps import runner_ps as runner
    elif args.with_cls:
        from tools.aqa import runner_aqa_clsloss as runner
    else:
        if multi_modality == 'late_fusion':
            from tools.aqa import runner_aqa_MM_late_fusion as runner
        elif multi_modality == 'early_fusion':
            if double_path == 'Double':
                from tools.aqa import runner_aqa_MM_early_fusion_double as runner
            else:
                from tools.aqa import runner_aqa_MM_early_fusion as runner
        else:
            from tools.aqa import runner_aqa as runner

    if args.test:
        runner.test_net(args)
    else:
        runner.train_net(args)
            

if __name__ == '__main__':
    main()