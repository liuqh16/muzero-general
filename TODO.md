1. 单智能体 -> 多智能体
    - MCTS进行rollout的改动：需要保存policy，因为每次固定一个智能体使用ucb进行search，其他智能体使用保存的policy进行sample
    - Env实际step的改动：原本直接使用MCTS得到的根节点policy进行greedy sample，现在和rollout的时候一样，也需要固定智能体，其他智能体使用网络policy直接sample
    - 问题：policy的滞后性很严重
2. 谷歌足球环境本身不是step-by-step的（例如围棋），而是同时决策，如何进行自博？
    - 固定先后顺序，例如先左边行动，后右边行动，再环境step更新
    - 一个思路：左先右后+左后右先，上下界bound
3. dynamic function需要输入action，muzero将其处理为one-hot向量，非常稀疏，且当动作空间变大时效果会不好
    - 网络结构修改：embeding层
    - gumbel
    - 检测不同动作的熵，将类似的动作合并

