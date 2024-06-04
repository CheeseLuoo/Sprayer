import copy
from .MCTS import MCTS
import concurrent.futures
from functools import partial

g_executor = concurrent.futures.ProcessPoolExecutor()

class MCTSConCurrent(MCTS):
  def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
      super().__init__(policy_value_fn, c_puct, n_playout)

  # state：board：MCTScontext
  def get_move(self, state):
    states = []
    for n in range(self._n_playout):
        # print("_n_playout:",n)
        states.append(state)
        state_copy = copy.deepcopy(state)
        self._playout(state_copy)
    # state_copy_list = list(map(copy.deepcopy, states))
    # partial_playout = partial(MCTSConCurrent._playout, self)
    # results = list(g_executor.map(partial_playout, state_copy_list))
    return max(self._root._children.items(),
            key=lambda act_node: act_node[1]._n_visits)[0]

  def _playout(self, state):
    super()._playout(state)
    # print("children")
    # print(len(self._root._children.items()))

# 每一个player都会实例化一个mcts，player不是指车辆，是一个玩游戏的上帝视角
class MCTSConcurrentPlayer(object):
  """AI player based on MCTS"""
  def __init__(self, policy_value_fn, c_puct=20, n_playout=500):
    self.mcts = MCTSConCurrent(policy_value_fn, c_puct, n_playout)

  def set_player_ind(self, p):
    self.player = p

  def reset_player(self):
    self.mcts.update_with_move(-1)

  # 将context传入，然后getmove
  def get_action(self, board):
    sensible_moves = board.availables
    if len(sensible_moves) > 0:
      move = self.mcts.get_move(board)
      self.mcts.update_with_move(-1)
      return move
    else:
      print("WARNING: the board is full")

  def __str__(self):
    return "MCTS {}".format(self.player)