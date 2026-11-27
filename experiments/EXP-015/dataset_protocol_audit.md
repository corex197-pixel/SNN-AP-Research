# 数据入口审查

官方 `main.py:187-192` 同时构造 CIFAR10 train=True 与 train=False 且 download=True；`:793-796` 每轮用测试集评价，`:838-852` 用 test_acc1 保存最佳；`:882-891` 重新加载后最终 test。本轮未调用官方 load_data 或官方入口。

`experiments/ROUND-014/controlled_data.py` 单独读取 5 个训练 data_batch，使用旧项目固定训练集索引，官方训练增强为 RandomCrop/RandomHorizontalFlip/CIFAR10Policy/Normalize/Cutout；验证时应使用原 4500 train-origin 索引和确定性变换，禁止影响训练状态。CPU smoke 读取了 2 个训练样本中的首个，记录见 controlled_data_test.json；未运行 4500 正式验证、未创建或读取官方 test_batch。还需在官方 CuPy 环境和完整模型上检验验证隔离。

官方 eval 用 model.eval/no_grad 并在每批 reset_net，不调用 calc_penalty；合成测试中的 state_dict 在 eval 前后不变。SparseBN 只有 training=True 才记录 pre_norm_input；若未清理则 eval 不覆盖旧缓存，必须显式隔离，CuPy 路径尚未验证。
