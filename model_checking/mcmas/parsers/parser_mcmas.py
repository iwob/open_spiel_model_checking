# from lark import Lark
#
# with open("grammar_ispl_v2.lark", encoding="utf-8") as f:
#     grammar = f.read()
#
# parser = Lark(
#     grammar,
#     parser="earley",
#     lexer="dynamic",
#     start="start",
# )
#
# with open("mnk_3,3,3_initial_U.ispl", encoding="utf-8") as f:
#     tree = parser.parse(f.read())
#
# print(tree.pretty())


from ispl_parser import parse_file

model = parse_file("mnk_3,3,3_initial_U.ispl")

# print(model)

print(model.semantics)

for agent in model.agents:
    print(agent.name)

print(model.environment.observable_vars)
print(model.formulae.formulas)