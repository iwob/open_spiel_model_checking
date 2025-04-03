from pathlib import Path
from absl import app
from absl import flags

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel
from stv.parsers.parser_stv_v2 import Stv2Parser, parseAndTransformFormula
from atl_model_game import AtlModelGame

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 10, "How often to print the exploitability")


def main(_):
  parser = Stv2Parser()
  file = Path(__file__).parent / "example_specifications" / "simple" / "simple.stv"
  with file.open() as f:
    text = f.read()
  stv_spec, formula = parser(text)
  game = AtlModelGame(stv_spec, formula)

  # CFR (its implementation in OpenSpiel?) requires sequential games, so let's change simultaneous game into sequential one
  game: pyspiel.Game = pyspiel.convert_to_turn_based(game)
  pyspiel.register_game(game.get_type(), AtlModelGame)
  print("Registered names:")
  print(pyspiel.registered_names())

  cfr_solver = cfr.CFRSolver(game)

  for i in range(FLAGS.iterations):
    cfr_solver.evaluate_and_update_policy()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.exploitability(game, cfr_solver.average_policy())
      print("Iteration {} exploitability {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)
