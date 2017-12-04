import tensorflow as tf
import numpy as np
import learn.ann as an
import learn

env=Environment(draw_r=-1,move_r=-0.001)
env.reset(state='6k1/8/8/5K2/8/8/8/3R4 w - -')
s='6k1/8/8/5K2/8/8/8/3R4 w - -'
f=featurize(chess.Board.from_epd(s)[0])
D=f.shape[1]
graph=ann.build_graph([128],D,learning_rate=0.0000001)
approx=DeepTDLambda(l=0.5,y=0.8,epochs=10,batch_sz=2,save_period=5)
pol=EpsilonGreedyPolicy(eps=1,decay_rate=0.00,decay_period=100)
sv=Supervisor(env,pol,approx,'test.sv',store_period=10,sample_period=10,mv_limit=33)

with tf.Session(graph=graph) as sess:
    with sess.as_default():
        saver=tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        fn=learn.model_path('init')
        saver.save(sess,learn.model_path(fn))
        sv.approx.set_modelfile('init')

        saver=tf.train.import_meta_graph(learn.meta_path(sv.approx.model_file))
        sess=learn.restore_session(saver,sv.approx.model_file) 

        with sess.as_default():
            updates=0
            sv.approx.set_session(sess)

            for _ in xrange(N):
                self.current_episode+=1
                rew, mv_cnt, win, update=self.run_episode(state=state)
                updates+=int(update)
