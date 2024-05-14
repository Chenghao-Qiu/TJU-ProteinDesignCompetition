import pandas as pd

# 读取Excel文件
excel_file_path = './GFP data.xlsx'  # 替换为您Excel文件的实际路径
sheet_name = 'avGFP,amacGFP,cgreGFP,ppluGFP2'  # 替换为包含数据的工作表名称
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# 选择所需的列并重命名以匹配微调代码中的列名
df = df[['aaMutations', 'GFP type', 'Brightness']]  # 替换为您Excel中的实际列名
df.columns = ['aaMutations', 'GFP type', 'Brightness']

# 保存为CSV文件
csv_file_path = 'GFP_row_data.csv'  # 您希望保存CSV文件的路径
df.to_csv(csv_file_path, index=False)

print(f'Data saved to {csv_file_path}')
