
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="IPG_CONF",
    settings_files=['settings.toml',  '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
