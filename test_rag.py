import asyncio
import os
from api.main import app
from modules import rag_system
from config.settings import settings

async def test_basic_query():
    """测试基本查询功能"""
    print("\n=== 测试基本查询 ===")
    
    # 测试问题
    test_questions = [
        "青少年抑郁症有哪些表现？",
        "如何帮助有自杀倾向的青少年？",
        "父母离异对青少年有什么影响？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        try:
            result = rag_system.query(question=question, age_group="teen", check_safety=True)
            print(f"回答: {result['answer'][:200]}...")
            print(f"安全检查: {'是' if result.get('is_crisis_response') else '否'}")
            print(f"来源数量: {len(result.get('sources', []))}")
        except Exception as e:
            print(f"错误: {e}")

async def test_safety_check():
    """测试安全检测功能"""
    print("\n=== 测试安全检测 ===")
    
    # 测试危机语句
    crisis_statements = [
        "我觉得活着没意思，想结束一切",
        "我在学校被同学欺负，每天都很痛苦",
        "我最近总是失眠，感觉很焦虑"
    ]
    
    for statement in crisis_statements:
        print(f"\n语句: {statement}")
        try:
            result = rag_system.query(question=statement, age_group="teen", check_safety=True)
            safety = result.get('safety_check') or {}
            print(f"危机等级: {safety.get('level', 'none')}")
            print(f"关键词: {[kw['keyword'] for kw in safety.get('keywords_found', [])]}")
            response = safety.get('safety_response') or {}
            if response.get('message'):
                print(f"安全响应: {response['message'][:100]}...")
        except Exception as e:
            print(f"错误: {e}")

async def main():
    """主测试函数"""
    print("开始测试青少年心理RAG系统...")
    
    # 验证配置
    settings.validate()
    print(f"配置验证通过 - 使用模型: {settings.CHAT_MODEL}")
    
    # 运行测试
    await test_basic_query()
    await test_safety_check()
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
