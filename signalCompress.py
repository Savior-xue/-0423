import numpy as np
from sklearn.decomposition import IncrementalPCA
from concurrent.futures import ThreadPoolExecutor
import os


class SignalCompressor:
    def __init__(self, n_sample, compress_rate=0.5, merge_channels=True):
        """
        改进版信号压缩处理器

        参数:
        n_sample: 每个通道的原始时域采样点数
        compress_rate: 压缩比率 (默认0.5)
        merge_channels: 是否合并双通道数据 (默认True)
        """
        self.n_sample = n_sample
        self.compress_rate = compress_rate
        self.merge_channels = merge_channels
        self.M = int(n_sample * compress_rate)
        self.H = None
        self.global_mean = None
        self.global_std = None
        self.is_trained = False

    def _batch_to_time(self, batch):
        x1_freq = batch[:, :self.n_sample]
        x2_freq = batch[:, self.n_sample:]

        # 添加转置操作（MATLAB的fft和Python的fft默认维度不同）
        x1_time = np.fft.irfft(x1_freq.T, n=self.n_sample).T  # (n_sample, batch) -> (batch, n_sample)
        x2_time = np.fft.irfft(x2_freq.T, n=self.n_sample).T
        return np.hstack([x1_time, x2_time])


    def _calculate_global_stats(self, X, batch_size, max_mem_gb=4):
        """计算全局统计量（带内存优化）"""
        # 内存限制计算
        elem_size = 4  # float32字节数
        max_samples = int(max_mem_gb * 1e9 / (2 * self.n_sample * elem_size))
        dynamic_batch = min(batch_size, max_samples)

        # 第一次遍历计算均值
        mean_accum = 0
        n_samples = 0
        for i in range(0, len(X), dynamic_batch):
            batch = X[i:i + dynamic_batch]
            time_data = self._batch_to_time(batch)
            n_samples += time_data.shape[0]
            mean_accum += np.sum(time_data, axis=0)
        self.global_mean = mean_accum / n_samples

        # 第二次遍历计算标准差
        std_accum = 0
        for i in range(0, len(X), dynamic_batch):
            batch = X[i:i + dynamic_batch]
            time_data = self._batch_to_time(batch)
            std_accum += np.sum((time_data - self.global_mean) ** 2, axis=0)
        self.global_std = np.sqrt(std_accum / n_samples)

    def fit(self, X_train, batch_size=1000, max_mem_gb=4):
        """使用PCA初始化压缩矩阵"""
        # 转换为时域数据
        time_data = self._batch_to_time(X_train)
        self.pca = IncrementalPCA(n_components=self.M)
        self.pca.fit(time_data)
        self.H = self.pca.components_.astype(np.float32)
        self.is_trained = True

    def _compress_batch(self, batch, n_workers=4):
        """多线程批量压缩"""

        def _process_subset(sub_batch):  # 移除self参数
            x1_freq = sub_batch[:, :self.n_sample]
            x2_freq = sub_batch[:, self.n_sample:]
            x1_time = np.fft.irfft(x1_freq, n=self.n_sample)
            x2_time = np.fft.irfft(x2_freq, n=self.n_sample)
            combined = np.hstack([x1_time, x2_time])
            return self.pca.transform(combined).astype(np.float32)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            results = []

            # 分片处理
            split_size = len(batch) // n_workers
            for i in range(n_workers):
                start = i * split_size
                end = (i + 1) * split_size if i < n_workers - 1 else len(batch)
                futures.append(executor.submit(_process_subset, batch[start:end]))

            for future in futures:
                results.append(future.result())

        return np.vstack(results).astype(np.float32)

    def save_compressed_data(self, X, y, save_dir=".", dataset_name="train", batch_size=1000, n_workers=4):
        """
        改进的保存方法（修正维度计算问题）
        """
        assert self.is_trained, "必须先训练压缩器！"
        os.makedirs(save_dir, exist_ok=True)

        # 根据通道合并情况确定输出维度
        output_dim = self.M if self.merge_channels else 2 * self.M  # 新增维度计算逻辑

        compressed_path = os.path.join(save_dir, f"X_{dataset_name}_compressed.dat")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

        compressed_shape = (len(X), output_dim)  # 使用动态计算的维度
        compressed_mmap = np.memmap(
            compressed_path,
            dtype=np.float32,
            mode='w+',
            shape=compressed_shape
        )

        # 多线程批处理
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            compressed = self._compress_batch(batch, n_workers)
            compressed_mmap[i:i + len(compressed)] = compressed

        # 保存标签
        np.save(os.path.join(save_dir, f"y_{dataset_name}.npy"), y)
        compressed_mmap.flush()

    # 模型保存/加载方法
    def save(self, path):
        """保存完整压缩模型"""
        # 新增目录创建逻辑
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        np.savez(path,
                 global_mean=self.global_mean,
                 global_std=self.global_std,
                 components=self.H)

    def load(self, path):
        """加载完整压缩模型"""
        data = np.load(path, allow_pickle=True)
        self.global_mean = data['global_mean']
        self.global_std = data['global_std']
        self.H = data['components']
        self.is_trained = True


if __name__ == "__main__":
    # 示例用法
    # 加载数据集
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")  # 新增测试集加载
    y_test = np.load("y_test.npy")

    # 初始化压缩器
    n_sample = 256
    compress_rate = 0.5
    compressor = SignalCompressor(n_sample=n_sample, compress_rate=compress_rate)

    # 训练压缩器
    compressor.fit(X_train)
    compressor.save("./compressed/compression_model.npz")  # 单独保存模型参数

    # 保存训练集压缩数据
    compressor.save_compressed_data(X_train, y_train,
                                    save_dir="./compressed",
                                    dataset_name="train",
                                    batch_size=2000,
                                    n_workers=4)

    # 新增：保存测试集压缩数据
    compressor.save_compressed_data(X_test, y_test,
                                    save_dir="./compressed",
                                    dataset_name="test",
                                    batch_size=2000,
                                    n_workers=4)

    # 验证训练集结果
    compressed_train = np.memmap('./compressed/X_train_compressed.dat',
                                dtype=np.float32,
                                shape=(len(X_train), 2 * compressor.M))

    # 验证测试集结果
    compressed_test = np.memmap('./compressed/X_test_compressed.dat',
                               dtype=np.float32,
                               shape=(len(X_test), 2 * compressor.M))

    print("\n训练集压缩验证:")
    print(f"压缩后维度: {compressed_train.shape}")
    print(f"存储节省: {(1 - compressed_train.nbytes/(X_train.nbytes/2)):.1%}")

    print("\n测试集压缩验证:")
    print(f"压缩后维度: {compressed_test.shape}")
    print(f"存储节省: {(1 - compressed_test.nbytes/(X_test.nbytes/2)):.1%}")