import json

# 读取源文件
with open("D3.json", "r", encoding="utf-8") as f:
    dialog_groups = json.load(f)

output_content = []

# 循环所有对话组，组数由文件内数组长度自动决定
for group_id, group_data in enumerate(dialog_groups, start=1):
    output_content.append(f"第{group_id}组：")
    message_arr = group_data["messages"]
    for message in message_arr:
        r = message["role"]
        c = message["content"]
        output_content.append(f"{r}：{c}")
    # 分组间隔空行
    output_content.append("")
    output_content.append("")

# 写入文本文件
with open("dialog_output.txt", "w", encoding="utf-8") as out_file:
    out_file.write("\n".join(output_content))

print(f"转换完成！一共识别到 {len(dialog_groups)} 组对话，已保存到 dialog_output.txt")