# -*- coding: utf-8 -*-
'''
Created on 2018年2月26日

@author: zwp
'''

'''
基础训练，构建基础模型，简单使用方法

单隐层 0-1 分类问题

'''



import tensorflow as tf;

class Model():
    W1 = None;# 输出层到隐层 权重 训练参数
    W2 = None;# 隐层到输出层 权重 训练参数
    b1 = None;# 输出层到隐层
    b2 = None;# 隐层到输出层 
    X = None; # 特征数据
    Y = None; # 标签数据 
    PY= None; # 预测标签值
    loss=None;# 损失值
    def __init__(self,feat_size,lab_size,hid_size,act_func=None):
        self.X = tf.placeholder(tf.float32, [None,feat_size], "X");
        self.Y = tf.placeholder(tf.float32, [None,lab_size], "Y");
        self.W1 = tf.Variable(tf.random_normal((feat_size,hid_size),dtype=tf.float32),"W1");
        self.W2 = tf.Variable(tf.random_normal((hid_size,lab_size),dtype=tf.float32),"W2");
        self.b1 = tf.Variable(tf.random_normal((hid_size,),dtype=tf.float32),"b1");
        self.b2 = tf.Variable(tf.random_normal((lab_size,),dtype=tf.float32),"b2");
        self.PY,self.loss = self.create_model(act_func);
        tf.global_variables_initializer();
        
    def create_model(self,act_func=None):
        if act_func==None:
            act_func = tf.sigmoid;
        H = act_func(tf.matmul(self.X,self.W1)+self.b1);
        PY = act_func(tf.matmul(H,self.W2)+self.b2);
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=PY, name="loss")); # 交叉熵 损失函数
        return PY,loss;
    
    
    def train(self,XS,YS,learn_rate,steps,batch_size,dat_size):
        train_setp = tf.train.AdamOptimizer(learn_rate).minimize(self.loss);
        saver = tf.train.Saver(tf.global_variables());
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer());
            for i in range(steps):
                start = (i*batch_size) % dat_size;
                end = min(start+batch_size,dat_size);
                _,py,loss,x,y=sess.run((train_setp,self.PY,self.loss,self.X,self.Y),{self.X:XS[start:end],self.Y:YS[start:end]})
#                 print(x,y);
                if i % 100 ==0:
                    print('step= %d '%(i),loss,py);
            print('\n finished! ',loss,py);      
            saver.save(sess, 'value_cache/save1.ckpt');
    def calculate(self,calX,calY):
        saver = tf.train.Saver(tf.global_variables());
        with tf.Session() as sess:
            saver.restore(sess, 'value_cache/save1.ckpt');
            loss,py=sess.run((self.loss,self.PY),{self.X:calX,self.Y:calY});
        return loss,py;


# end class Model

xs =[
        [0.1,0.1],
        [0.2,0.2],
        [0.1,0.2],
        [0.2,0.1],
        
        [-0.1,0.1],
        [-0.2,0.2],
        [-0.1,0.2],
        [-0.2,0.1],

        [-0.1,-0.1],
        [-0.2,-0.2],
        [-0.1,-0.2],
        [-0.2,-0.1],

        [0.1,-0.1],
        [0.2,-0.2],
        [0.1,-0.2],
        [0.2,-0.1]
    ];
txs=[
    [0.15,0.15],
    [-0.08,0.3],
    [0.25,0.09],
    [0.0,0.1],
#     [0.1,0.19],
#     [-0.15,0.15],
#     [0.15,-0.15],
#     [-0.15,-0.15],
#     [0.15,0.15],
#     [-0.15,0.15],
#     [0.15,-0.15],
#     [-0.15,-0.15],
#     [0.15,0.15],
#     [-0.15,0.15],
#     [0.15,-0.15],
#     [-0.15,-0.15]                
    ];
ys =[
    [1,0],[1,0],[1,0],[1,0],
    [0,1],[0,1],[0,1],[0,1],
    [1,0],[1,0],[1,0],[1,0],
    [0,1],[0,1],[0,1],[0,1]
    ];
tys =[
    [1,0],[1,0],[1,0],[1,0],
#     [0,1],[0,1],[0,1],[0,1],
#     [1,0],[1,0],[1,0],[1,0],
#     [0,1],[0,1],[0,1],[0,1]
    ];     

# 特征数
feature_size = 2;
# 标签维度
label_size = 2;
# 隐层数
hidden_size = 4;

# 激活函数
def act_func(X):
    return tf.sigmoid(X);

learn_rate = 0.1;
steps = 1000;
batch_size = 16;
dat_size=16;

def run():

    simple=Model(feature_size,label_size,hidden_size,act_func);
    
    simple.train(xs, ys, learn_rate, steps, batch_size, dat_size)
    
    print(simple.calculate(txs, tys));
    pass;


if __name__ == '__main__':
    run();
    
    pass