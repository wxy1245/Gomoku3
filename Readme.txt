# How to use it: run human_ai_visualize.py to be a player, or you can also see the ai plays against each other

My Version of Gomoku, codes mainly from Song JunXiao

1. !!!!!! use the prior knowledge that the last moves must be more relevant to the result(win or lose)

2. Use the saved best model as opponent, so it is no need to use mctsPure as a rival.

3. To help loss reduce, I demand that after 3 evaluation rounds of tie, increase the caculation resource(data buffer size, so can random choose in a larger scope) and ability(playouts num).

4. abandon the origin theano, only keep the pytorch code

5. !!!!!! Add human vs_ai games record

(6. Consider the last move decides who win may reduce calculation, but actually it seems to be invalid)

note: For my version, AI may be better at defense. You can see the gif to notice that Song's version tend to get own victory even as a afterplayer, but you can see in my record pictures, my leafDamping AI tends to prevent firstplayer's victory.

By the way, to move a step forward, I think it is helpful to add some high-quality data of human top players, used as a pretraining. Or, increase a small ratio of data between our model played with a already known strong ai player. 