import pandas as pd

# 原始蛋白质序列字典
protein_sequences = {
    'avGFP': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
    'amacGFP': 'MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNDFFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMDFKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQTNVPLGDGPVLIPINHYLSCQTAISKDRNETRDHMVFLEFFSACGHTHGMDELYK',
    'cgreGFP': 'MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR',
    'ppluGFP': 'MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA'
}

# 定义一个函数来应用突变


def apply_mutations(sequence, mutations):
    # 如果突变标记为WT，则返回原始序列
    if mutations == 'WT':
        return sequence

    # 将序列转换为列表，以便我们可以修改它
    sequence_list = list(sequence)

    # 分割突变字符串并遍历每个突变
    for mutation in mutations.split(':'):
        # 确保突变字符串不为空
        if mutation:
            # 突变的位置是从1开始的，所以我们需要减去1来得到正确的索引
            index = int(mutation[1:-1]) - 1
            # 突变后的氨基酸
            new_aa = mutation[-1]
            # 应用突变
            sequence_list[index] = new_aa

    # 将列表转换回字符串
    return ''.join(sequence_list)

# 读取CSV文件
csv_file_path = './GFP_row_data.csv'  # 替换为您CSV文件的实际路径
df = pd.read_csv(csv_file_path)

# 应用突变到原始序列
df['Full Sequence'] = df.apply(lambda row: apply_mutations(
    protein_sequences.get(row['GFP type'], ''), row['aaMutations']), axis=1)

# 选择所需的列并重命名以匹配微调代码中的列名
df = df[['Full Sequence', 'GFP type', 'Brightness']]

# 保存为新的CSV文件
new_csv_file_path = 'GFP_data_with_full_sequences.csv'  # 您希望保存新CSV文件的路径
df.to_csv(new_csv_file_path, index=False)

print(f'Data saved to {new_csv_file_path}')
