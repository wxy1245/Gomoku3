My Version of Gomoku, codes mainly from Song JunXiao

1. !!!!!! use the prior knowledge that the last moves must be more relevant to the result(win or lose)

2. Use the saved best model as opponent, so it is no need to use mctsPure as a rival.

3. abandon the origin theano, only keep the pytorch code

(4. Consider the last move decides who win may reduce calculation, but actually it seems to be invalid)

note: For my version, AI is better at defense. You can see the gif to notice that Song's version tend to get own victory even as a afterplayer, but you can see in my record pictures, my leafDamping AI tends to prevent firstplayer's victory.