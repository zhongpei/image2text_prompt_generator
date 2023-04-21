import torch.cuda
import torch.backends

from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="IPG_CONF",
    settings_files=['settings.toml', '.secrets.toml'],
)


