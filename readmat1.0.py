import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class DataProcessor:
    def __init__(self, mat_path, batch_size=2000, test_size=0.2, max_workers=4, random_state=42):
        """
        初始化数据处理器

        参数:
        mat_path:       数据集路径
        batch_size:     批处理大小（默认2000）
        test_size:      测试集比例（默认0.2）
        max_workers:    最大并行工作线程数（默认4）
        random_state:   随机种子（默认42）
        """
        self.mat_path = mat_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.max_workers = max_workers
        self.random_state = random_state

        # 初始化运行时参数
        self.n_sample = None  # 每个通道的采样点数
        self.total_samples = None  # 总样本数
        self.feature_dim = None  # 特征维度
        self.mmap_array = None  # 内存映射数组
        self.labels = None  # 数据标签
        self.mmap_path = os.path.join(os.getcwd(), 'dataset.mmap')  # 内存映射文件路径

        # 数据集划分结果
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def process_data(self):
        """主处理流程"""
        # 初始化HDF5文件参数
        self._get_dataset_metadata()

        # 准备内存映射文件
        self._prepare_memmap()

        # 并行处理数据
        self._parallel_processing()

        # 加载标签数据
        self._load_labels()

    def _get_dataset_metadata(self):
        """获取数据集元数据"""
        with h5py.File(self.mat_path, 'r') as f:
            # 获取原始数据维度信息
            real_shape = f['Data']['real'].shape
            self.n_sample = real_shape[0] // 2  # 每个通道的采样点数
            self.total_samples = real_shape[1]  # 总样本数
            self.feature_dim = self.n_sample * 2  # 频域特征维度

    def _prepare_memmap(self):
        """准备内存映射文件"""
        if os.path.exists(self.mmap_path):
            os.remove(self.mmap_path)

        # 初始化复数格式的内存映射
        self.mmap_array = np.memmap(
            self.mmap_path,
            dtype=np.complex64,
            mode='w+',
            shape=(self.total_samples, self.feature_dim)
        )

    def _parallel_processing(self):
        """并行处理数据"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            # 提交批处理任务
            for batch, start, end in self._chunk_generator():
                future = executor.submit(
                    self._process_batch,
                    batch, start, end
                )
                futures.append(future)

            # 等待所有任务完成
            for future in as_completed(futures):
                future.result()

    def _chunk_generator(self):
        """数据分块生成器"""
        with h5py.File(self.mat_path, 'r') as f:
            real_part = f['Data']['real'][:]
            imag_part = f['Data']['imag'][:]

            for start in range(0, self.total_samples, self.batch_size):
                end = min(start + self.batch_size, self.total_samples)
                # 构造复数数据并转置为（样本数×特征数）
                complex_data = (real_part[:, start:end].T + 1j * imag_part[:, start:end].T)
                yield complex_data.astype(np.complex64), start, end

    def _process_batch(self, batch, start, end):
        """处理单个数据批次"""
        # 时域转频域
        freq_data = self._to_frequency_domain(batch)

        # 执行标准化
        normalized = self._normalize_complex(freq_data)

        # 写入内存映射文件
        self.mmap_array[start:end] = normalized

    def _to_frequency_domain(self, time_data):
        """时域转频域处理"""
        x1 = time_data[:, :self.n_sample]  # 第一通道
        x2 = time_data[:, self.n_sample:]  # 第二通道

        # 执行FFT变换
        X1 = np.fft.fft(x1, axis=1)
        X2 = np.fft.fft(x2, axis=1)

        return np.hstack((X1, X2))

    def _normalize_complex(self, complex_data):
        """复数数据标准化（实部虚部分别处理）"""
        real_part = complex_data.real
        imag_part = complex_data.imag

        # 分别计算统计量
        real_mean, real_std = np.mean(real_part), np.std(real_part)
        imag_mean, imag_std = np.mean(imag_part), np.std(imag_part)

        # 执行标准化
        normalized_real = (real_part - real_mean) / real_std
        normalized_imag = (imag_part - imag_mean) / imag_std

        return normalized_real + 1j * normalized_imag

    def _load_labels(self):
        """加载标签数据"""
        with h5py.File(self.mat_path, 'r') as f:
            self.labels = f['Label'][:].flatten().astype(np.float32)

    def split_dataset(self):
        """划分训练集和测试集"""
        indices = np.arange(self.total_samples)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_state
        )

        self.X_train = self.mmap_array[train_idx]
        self.X_test = self.mmap_array[test_idx]
        self.y_train = self.labels[train_idx]
        self.y_test = self.labels[test_idx]

    def save_results(self, save_dir="."):
        """保存处理结果"""
        np.save(os.path.join(save_dir, "X_train.npy"), self.X_train)
        np.save(os.path.join(save_dir, "X_test.npy"), self.X_test)
        np.save(os.path.join(save_dir, "y_train.npy"), self.y_train)
        np.save(os.path.join(save_dir, "y_test.npy"), self.y_test)

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.mmap_path):
            os.remove(self.mmap_path)


if __name__ == "__main__":
    # 使用示例
    processor = DataProcessor(mat_path="Dataset.mat")

    # 执行数据处理流程
    processor.process_data()

    # 划分数据集
    processor.split_dataset()

    # 保存结果
    processor.save_results()

    # 清理临时文件（可选）
    # processor.cleanup()

    # 验证结果
    print("\n处理结果验证:")
    print(f"训练集形状: {processor.X_train.shape}")
    print(f"测试集形状: {processor.X_test.shape}")
    print("特征范围（实部）: [{:.2f}, {:.2f}]".format(
        np.min(processor.X_train.real),
        np.max(processor.X_train.real)
    ))
    print("特征范围（虚部）: [{:.2f}, {:.2f}]".format(
        np.min(processor.X_train.imag),
        np.max(processor.X_train.imag)
    ))