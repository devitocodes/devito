from devito import warning

warning("""The location of Devito's checkpointing has changed. This location will be
           deprecated soon. Please change your imports to 'from devito import
           DevitoCheckpoint, CheckpointOperator'""")

from devito.checkpointing import *  # noqa
