# How to use it: run human_ai_visualize.py to be a player, or you can also see the ai plays against each other

My Version of Gomoku, codes mainly from Song JunXiao

1. !!!!!! use the prior knowledge that the last moves must be more relevant to the result(win or lose), so give higher reward to last moves.

2. Use the saved best model as opponent, so it is no need to use mctsPure as a rival.

3. To help loss reduce, I demand that after 3 evaluation rounds of tie, increase the caculation resource(data buffer size, so can random choose in a larger scope) and ability(playouts num).

4. abandon the origin theano, only keep the pytorch code

5. !!!!!! Add human vs_ai games record
Increase the data between our model played with a already known strong ai player, like me, would be helpful. The Human AI play data help improve the model, as you can see in the Human AI advanced model(v0, v1, v2).
Also, I plan to use it on a Gomoku app, let it play against human players, collect the lose data, and may help it improve and eventually defeat human players.