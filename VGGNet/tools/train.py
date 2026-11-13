import argparse

import compat  
import vggnet.backbones.vgg  
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.launcher = args.launcher

    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
