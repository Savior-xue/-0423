import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class DataProcessor:
    def __init__(self, mat_path, batch_size=1000, test_size=0.2, max_workers=4,
                 random_state=42, num_augments=1, noise_level=0.05):
        """初始化数据处理器"""
        self.mat_path = mat_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.max_workers = max_workers
        self.random_state = random_state
        self.num_augments = num_augments
        self.noise_level = noise_level

        # 初始化运行时参数
        self.n_sample = None
        self.total_samples = None
        self.feature_dim = None
        self.mmap_array = None
        self.labels = None
        self.mmap_path = os.path.join(os.getcwd(), 'dataset.mmap')

        # 数据集划分结果
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def process_data(self):
        """主处理流程"""
        self._get_dataset_metadata()
        self._prepare_memmap()
        self._parallel_processing()
        self._load_labels()

    def _get_dataset_metadata(self):
        """获取数据集元数据"""
        with h5py.File(self.mat_path, 'r') as f:
            real_shape = f['Data']['real'].shape
            self.n_sample = real_shape[0] // 2
            self.total_samples = real_shape[1]
            self.feature_dim = self.n_sample * 2

    def _prepare_memmap(self):
        """准备内存映射文件"""
        if os.path.exists(self.mmap_path):
            os.remove(self.mmap_path)
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
            for batch, start, end in self._chunk_generator():
                future = executor.submit(self._process_batch, batch, start, end)
                futures.append(future)
            for future in as_completed(futures):
                future.result()

    def _chunk_generator(self):
        """数据分块生成器"""
        with h5py.File(self.mat_path, 'r') as f:
            real_part = f['Data']['real'][:]
            imag_part = f['Data']['imag'][:]
            for start in range(0, self.total_samples, self.batch_size):
                end = min(start + self.batch_size, self.total_samples)
                complex_data = (real_part[:, start:end].T + 1j * imag_part[:, start:end].T)
                yield complex_data.astype(np.complex64), start, end

    def _process_batch(self, batch, start, end):
        """处理单个数据批次"""
        freq_data = self._to_frequency_domain(batch)
        normalized = self._normalize_complex(freq_data)
        self.mmap_array[start:end] = normalized

    def _to_frequency_domain(self, time_data):
        """时域转频域处理"""
        x1 = time_data[:, :self.n_sample]
        x2 = time_data[:, self.n_sample:]
        X1 = np.fft.fft(x1, axis=1)
        X2 = np.fft.fft(x2, axis=1)
        return np.hstack((X1, X2))

    def _normalize_complex(self, complex_data):
        """复数数据标准化"""
        real_part = complex_data.real
        imag_part = complex_data.imag
        real_mean, real_std = np.mean(real_part), np.std(real_part)
        imag_mean, imag_std = np.mean(imag_part), np.std(imag_part)
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
            indices, test_size=self.test_size,
            shuffle=True, random_state=self.random_state
        )
        self.X_train = self.mmap_array[train_idx]
        self.X_test = self.mmap_array[test_idx]
        self.y_train = self.labels[train_idx]
        self.y_test = self.labels[test_idx]

        # 数据增强和归一化
        self._data_augmentation()
        self._apply_normalization()

    def _data_augmentation(self):
        """优化后的数据增强方法"""
        original_samples = len(self.X_train)
        total_augments = 2 * self.num_augments  # 每次增强产生两种样本
        total_augmented_samples = original_samples * (1 + total_augments)

        # 创建内存映射文件存储增强数据
        aug_data_path = os.path.join(os.getcwd(), 'aug_data.mmap')
        if os.path.exists(aug_data_path):
            os.remove(aug_data_path)
        aug_data = np.memmap(aug_data_path, dtype=np.complex64, mode='w+',
                             shape=(total_augmented_samples, self.feature_dim))

        # 创建内存映射文件存储增强标签
        aug_label_path = os.path.join(os.getcwd(), 'aug_labels.mmap')
        if os.path.exists(aug_label_path):
            os.remove(aug_label_path)
        aug_labels = np.memmap(aug_label_path, dtype=np.float32, mode='w+',
                               shape=(total_augmented_samples,))

        # 复制原始数据
        aug_data[:original_samples] = self.X_train
        aug_labels[:original_samples] = self.y_train
        current_pos = original_samples

        # 分批次增强
        batch_size = 1000  # 根据内存调整
        for i in range(0, original_samples, batch_size):
            batch_end = min(i + batch_size, original_samples)
            data_batch = self.X_train[i:batch_end]
            label_batch = self.y_train[i:batch_end]
            batch_length = len(data_batch)

            for _ in range(self.num_augments):
                # 噪声增强
                noise = (np.random.normal(0, self.noise_level, data_batch.shape) +
                         1j * np.random.normal(0, self.noise_level, data_batch.shape))
                aug_data[current_pos:current_pos + batch_length] = data_batch + noise
                aug_labels[current_pos:current_pos + batch_length] = label_batch
                current_pos += batch_length

                # 相位旋转
                phase = np.exp(1j * np.random.uniform(-np.pi / 8, np.pi / 8, batch_length))
                aug_data[current_pos:current_pos + batch_length] = data_batch * phase[:, np.newaxis]
                aug_labels[current_pos:current_pos + batch_length] = label_batch
                current_pos += batch_length

        # 更新训练数据引用
        self.X_train = aug_data
        self.y_train = aug_labels

    def _apply_normalization(self):
        """优化后的归一化方法，使用内存映射分块处理"""
        real_scaler = MinMaxScaler()
        imag_scaler = MinMaxScaler()

        # 使用样本数据进行拟合
        sample_idx = np.random.choice(len(self.X_train),
                                      size=min(10000, len(self.X_train)),
                                      replace=False)
        sample_real = self.X_train[sample_idx].real.astype(np.float32)
        sample_imag = self.X_train[sample_idx].imag.astype(np.float32)
        real_scaler.fit(sample_real)
        imag_scaler.fit(sample_imag)

        # 创建归一化后的内存映射文件
        def create_normalized_memmap(data, path):
            if os.path.exists(path):
                os.remove(path)
            normalized_mmap = np.memmap(
                path,
                dtype=np.complex64,
                mode='w+',
                shape=data.shape
            )

            # 分块处理数据
            batch_size = 5000
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                # 处理实部和虚部
                real_part = real_scaler.transform(batch.real.astype(np.float32))
                imag_part = imag_scaler.transform(batch.imag.astype(np.float32))
                normalized_mmap[i:i + batch_size] = real_part + 1j * imag_part
                normalized_mmap.flush()  # 确保写入磁盘
            return normalized_mmap

        # 处理训练数据
        train_norm_path = os.path.join(os.getcwd(), 'norm_train.mmap')
        self.X_train = create_normalized_memmap(self.X_train, train_norm_path)


    def save_results(self, save_dir="."):
        """保存处理结果"""

        # 分块保存训练数据
        def save_in_batches(data, path):
            with open(path, 'wb') as f:
                np.save(f, data[:0])  # 创建空文件
                for i in range(0, len(data), 100000):
                    batch = data[i:i + 100000]
                    with open(path, 'ab') as f:
                        np.save(f, batch)

        save_in_batches(self.X_train, os.path.join(save_dir, "X_train.npy"))
        save_in_batches(self.X_test, os.path.join(save_dir, "X_test.npy"))
        np.save(os.path.join(save_dir, "y_train.npy"), self.y_train)
        np.save(os.path.join(save_dir, "y_test.npy"), self.y_test)

    def cleanup(self):
        """清理所有临时文件"""
        temp_files = [
            self.mmap_path,
            'aug_data.mmap',
            'aug_labels.mmap',
            'norm_data.mmap'
        ]
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    processor = DataProcessor(mat_path="Dataset.mat", num_augments=1)
    processor.process_data()
    processor.split_dataset()
    processor.save_results()
    processor.cleanup()

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
