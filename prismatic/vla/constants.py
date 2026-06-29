import sys
from enum import Enum
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 151386
STOP_INDEX = 2
NUM_TOKENS = 64

class NormalizationType(str, Enum):
    NORMAL = 'normal'
    BOUNDS = 'bounds'
    BOUNDS_Q99 = 'bounds_q99'
LIBERO_CONSTANTS = {'NUM_ACTIONS_CHUNK': 8, 'ACTION_DIM': 7, 'PROPRIO_DIM': 8, 'ACTION_PROPRIO_NORMALIZATION_TYPE': NormalizationType.BOUNDS_Q99}
CALVIN_CONSTANTS = {'NUM_ACTIONS_CHUNK': 8, 'ACTION_DIM': 7, 'PROPRIO_DIM': 8, 'ACTION_PROPRIO_NORMALIZATION_TYPE': NormalizationType.BOUNDS_Q99}
ALOHA_CONSTANTS = {'NUM_ACTIONS_CHUNK': 25, 'ACTION_DIM': 14, 'PROPRIO_DIM': 14, 'ACTION_PROPRIO_NORMALIZATION_TYPE': NormalizationType.BOUNDS}
BRIDGE_CONSTANTS = {'NUM_ACTIONS_CHUNK': 5, 'ACTION_DIM': 7, 'PROPRIO_DIM': 7, 'ACTION_PROPRIO_NORMALIZATION_TYPE': NormalizationType.BOUNDS_Q99}

def detect_robot_platform():
    cmd_args = ' '.join(sys.argv).lower()
    if 'libero' in cmd_args:
        return 'LIBERO'
    elif 'aloha' in cmd_args:
        return 'ALOHA'
    elif 'bridge' in cmd_args:
        return 'BRIDGE'
    elif 'calvin' in cmd_args:
        return 'CALVIN'
    else:
        return 'LIBERO'
ROBOT_PLATFORM = detect_robot_platform()
if ROBOT_PLATFORM == 'LIBERO':
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == 'ALOHA':
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == 'BRIDGE':
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == 'CALVIN':
    constants = CALVIN_CONSTANTS
NUM_ACTIONS_CHUNK = constants['NUM_ACTIONS_CHUNK']
ACTION_DIM = constants['ACTION_DIM']
PROPRIO_DIM = constants['PROPRIO_DIM']
ACTION_PROPRIO_NORMALIZATION_TYPE = constants['ACTION_PROPRIO_NORMALIZATION_TYPE']
print(f'Using {ROBOT_PLATFORM} constants:')
print(f'  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}')
print(f'  ACTION_DIM = {ACTION_DIM}')
print(f'  PROPRIO_DIM = {PROPRIO_DIM}')
print(f'  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}')
print('If needed, manually set the correct constants in `prismatic/vla/constants.py`!')
