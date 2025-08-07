from utils import calculate_f1_score_from_csv
import os

if __name__ == '__main__':
    # --- 使用示例 ---

    # 1. 创建一个示例CSV文件
    csv_data = """line_number,TP,FP,FN,model_output_valid,field_项目名称,field_公告类型,field_行业分类,field_发布时间,field_预算金额,field_采购人,field_响应文件截至提交时间,field_开标地址,field_所在地区
    1,8,1,1,True,OK (Matched),OK (Matched),Mismatch,OK (Matched),OK (Matched),OK (Matched),OK (Matched),OK (Matched),OK (Matched)
    2,9,0,0,True,OK (Matched),OK (Matched),OK (Matched),OK (Matched),OK (Matched),OK (Matched),Mismatch,OK (Matched),OK (Matched)
    """

    file_name = "temp_f1_score_data.csv"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(csv_data)

    # 2. 调用函数并传入文件路径
    f1_score_result = calculate_f1_score_from_csv(file_name)

    # 3. 打印结果
    print(f"\n计算出的 F1-Score 为: {f1_score_result:.4f}")

    # 4. (可选) 清理创建的临时文件
    os.remove(file_name)
