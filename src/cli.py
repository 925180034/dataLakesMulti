#!/usr/bin/env python3
"""
数据湖多智能体系统命令行界面
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import click
from src.core.multi_agent_workflow import create_workflow, DataLakeDiscoveryWorkflow
from src.core.models import AgentState, TableInfo, ColumnInfo
from src.config.settings import settings
from src.utils.data_parser import parse_tables_data, parse_columns_data


@click.group()
@click.version_option(version=settings.version)
def cli():
    """数据湖多智能体系统 CLI"""
    # 创建必要的目录
    settings.create_directories()


@cli.command()
@click.option('--query', '-q', required=True, help='用户查询语句')
@click.option('--tables', '-t', help='查询表的JSON文件路径')
@click.option('--columns', '-c', help='查询列的JSON文件路径') 
@click.option('--output', '-o', help='输出结果到文件')
@click.option('--format', '-f', type=click.Choice(['json', 'markdown', 'table']), default='markdown', help='输出格式')
@click.option('--all-tables', help='所有表的JSON文件路径（用于初始化优化工作流）')
@click.option('--no-optimize', is_flag=True, help='禁用优化工作流，使用基础版本')
def discover(query: str, tables: str, columns: str, output: str, format: str, all_tables: str = None, no_optimize: bool = False):
    """执行数据发现"""
    try:
        # 加载输入数据
        query_tables = []
        query_columns = []
        
        if tables:
            tables_path = Path(tables)
            if not tables_path.exists():
                click.echo(f"错误: 表文件不存在: {tables}", err=True)
                sys.exit(1)
            
            with open(tables_path) as f:
                tables_data = json.load(f)
                query_tables = parse_tables_data(tables_data)
        
        if columns:
            columns_path = Path(columns)
            if not columns_path.exists():
                click.echo(f"错误: 列文件不存在: {columns}", err=True)
                sys.exit(1)
            
            with open(columns_path) as f:
                columns_data = json.load(f)
                query_columns = parse_columns_data(columns_data)
        
        if not query_tables and not query_columns:
            click.echo("错误: 必须提供 --tables 或 --columns 参数", err=True)
            sys.exit(1)
        
        # 加载所有表数据（如果提供）
        all_tables_data = None
        if all_tables:
            if format != 'json':
                click.echo(f"📊 加载所有表数据: {all_tables}")
            all_tables_path = Path(all_tables)
            if not all_tables_path.exists():
                click.echo(f"错误: 找不到文件 {all_tables}", err=True)
                sys.exit(1)
            
            with open(all_tables_path) as f:
                all_tables_data = json.load(f)
                if format != 'json':
                    click.echo(f"✅ 已加载 {len(all_tables_data)} 个表")
        
        # 执行发现 - 只在非JSON格式时显示进度
        if format != 'json':
            if no_optimize:
                click.echo("🔍 开始数据发现（使用基础工作流）...")
            else:
                click.echo("🚀 开始数据发现（使用优化工作流）...")
        
        result = asyncio.run(discover_data(
            user_query=query,
            query_tables=[t.model_dump() for t in query_tables],
            query_columns=[c.model_dump() for c in query_columns],
            all_tables_data=all_tables_data,
            use_optimized=not no_optimize
        ))
        
        # 确保result是AgentState对象
        if isinstance(result, dict):
            from src.core.models import AgentState
            result = AgentState.from_dict(result)
        
        # 格式化输出
        if format == 'json':
            output_text = _format_json_output(result)
        elif format == 'table':
            output_text = _format_table_output(result)
        else:  # markdown
            output_text = _format_markdown_output(result)
        
        # 输出结果
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            if format != 'json':  # JSON格式时避免额外输出
                click.echo(f"✅ 结果已保存到: {output}")
        else:
            if format == 'json':
                # JSON格式时只输出纯JSON，不添加任何前缀或后缀
                click.echo(output_text)
            else:
                # 非JSON格式时直接输出内容（进度信息已在上面显示）
                click.echo(output_text)
        
        # 只在非JSON格式时显示摘要
        if format != 'json':
            if result.final_results:
                click.echo(f"\n📊 摘要: 找到 {len(result.final_results)} 个匹配结果")
            else:
                click.echo("\n❌ 未找到匹配结果")
            
    except Exception as e:
        click.echo(f"❌ 发现失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--tables', '-t', required=True, help='表数据JSON文件路径')
@click.option('--columns', '-c', help='列数据JSON文件路径（可选）')
@click.option('--output', '-o', help='索引保存路径（可选）')
def build_index(tables: str, columns: str, output: str):
    """构建WebTable数据索引（推荐使用）"""
    try:
        from src.tools.data_indexer import build_webtable_indices
        
        tables_path = Path(tables)
        if not tables_path.exists():
            click.echo(f"错误: 表文件不存在: {tables}", err=True)
            sys.exit(1)
        
        if columns:
            columns_path = Path(columns)
            if not columns_path.exists():
                click.echo(f"错误: 列文件不存在: {columns}", err=True)
                sys.exit(1)
        
        click.echo("🔧 开始构建WebTable数据索引...")
        click.echo(f"📊 表数据文件: {tables}")
        if columns:
            click.echo(f"📊 列数据文件: {columns}")
        if output:
            click.echo(f"💾 索引保存路径: {output}")
        
        # 构建索引
        result = asyncio.run(build_webtable_indices(
            tables_file=str(tables_path),
            columns_file=str(columns_path) if columns else None,
            save_path=output
        ))
        
        if result['status'] == 'success':
            click.echo("\n✅ 索引构建完成!")
            click.echo(f"📊 统计信息:")
            click.echo(f"  - 处理表数: {result['tables_processed']}")
            click.echo(f"  - 索引表数: {result['tables_indexed']}")
            click.echo(f"  - 处理列数: {result['columns_processed']}")
            click.echo(f"  - 索引列数: {result['columns_indexed']}")
            click.echo(f"  - 索引路径: {result['index_path']}")
        else:
            click.echo(f"❌ 索引构建失败: {result.get('error', '未知错误')}", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ 索引构建失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--tables', '-t', required=True, help='表数据JSON文件路径')
def index_tables(tables: str):
    """为表建立索引（兼容性命令）"""
    try:
        tables_path = Path(tables)
        if not tables_path.exists():
            click.echo(f"错误: 文件不存在: {tables}", err=True)
            sys.exit(1)
        
        with open(tables_path) as f:
            tables_data = json.load(f)
        
        table_infos = [TableInfo(**data) for data in tables_data]
        
        click.echo(f"🔧 开始为 {len(table_infos)} 个表建立索引...")
        
        workflow = create_workflow()
        asyncio.run(workflow.table_discovery.initialize_table_index(table_infos))
        
        click.echo("✅ 表索引建立完成")
        
    except Exception as e:
        click.echo(f"❌ 索引建立失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--columns', '-c', required=True, help='列数据JSON文件路径')
def index_columns(columns: str):
    """为列建立索引（兼容性命令）"""
    try:
        columns_path = Path(columns)
        if not columns_path.exists():
            click.echo(f"错误: 文件不存在: {columns}", err=True)
            sys.exit(1)
        
        with open(columns_path) as f:
            columns_data = json.load(f)
        
        column_infos = [ColumnInfo(**data) for data in columns_data]
        
        click.echo(f"🔧 开始为 {len(column_infos)} 个列建立索引...")
        
        workflow = create_workflow()
        asyncio.run(workflow.column_discovery.initialize_indices(column_infos))
        
        click.echo("✅ 列索引建立完成")
        
    except Exception as e:
        click.echo(f"❌ 索引建立失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--path', '-p', help='索引路径（可选，默认使用配置路径）')
def verify_index(path: str):
    """验证索引是否正确构建"""
    try:
        from src.tools.data_indexer import verify_indices
        
        click.echo("🔍 开始验证索引...")
        
        result = asyncio.run(verify_indices(path))
        
        if result['status'] == 'success':
            click.echo("✅ 索引验证成功!")
            click.echo("📊 索引统计:")
            
            vector_stats = result.get('vector_search', {})
            click.echo(f"  向量搜索:")
            click.echo(f"    - 列数量: {vector_stats.get('column_count', 0)}")
            click.echo(f"    - 表数量: {vector_stats.get('table_count', 0)}")
            
            value_stats = result.get('value_search', {})
            click.echo(f"  值搜索:")
            click.echo(f"    - 索引列数: {value_stats.get('indexed_columns', 0)}")
            
            if vector_stats.get('column_count', 0) == 0 and vector_stats.get('table_count', 0) == 0:
                click.echo("\n⚠️  警告: 索引为空，请先使用 build-index 命令构建索引")
        else:
            click.echo(f"❌ 索引验证失败: {result.get('error', '未知错误')}", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ 索引验证失败: {e}", err=True)
        sys.exit(1)


@cli.command()
def serve():
    """启动API服务"""
    try:
        import uvicorn
        from src.api import app
        
        click.echo(f"🚀 启动API服务 (端口: 8000)...")
        click.echo(f"📖 API文档: http://localhost:8000/docs")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level=settings.logging.level.lower()
        )
        
    except ImportError:
        click.echo("❌ 需要安装 uvicorn: pip install uvicorn", err=True)
        sys.exit(1)
    except Exception as e: 
        click.echo(f"❌ 服务启动失败: {e}", err=True)
        sys.exit(1)


@cli.command()
def config():
    """显示当前配置"""
    config_info = {
        "项目名称": settings.project_name,
        "版本": settings.version,
        "调试模式": settings.debug,
        "LLM提供商": settings.llm.provider,
        "LLM模型": settings.llm.model_name,
        "向量数据库": settings.vector_db.provider,
        "数据目录": str(settings.data_dir),
        "缓存目录": str(settings.cache_dir),
        "日志目录": str(settings.log_dir)
    }
    
    click.echo("⚙️  当前配置:")
    for key, value in config_info.items():
        click.echo(f"  {key}: {value}")


@cli.command()
def api_status():
    """显示API密钥状态"""
    try:
        from src.utils.llm_client import llm_client
        
        # 检查LLM客户端是否支持API状态查询
        if hasattr(llm_client, 'get_api_status'):
            status = llm_client.get_api_status()
            
            click.echo("🔑 API密钥状态:")
            click.echo(f"  提供商: {status.get('provider', 'unknown')}")
            click.echo(f"  总密钥数: {status.get('total_keys', 0)}")
            click.echo(f"  可用密钥数: {status.get('active_keys', 0)}")
            
            # 显示每个密钥的详细状态
            if 'keys_status' in status:
                click.echo("\n密钥详情:")
                for i, key_info in enumerate(status['keys_status'], 1):
                    status_icon = "✅" if key_info['is_available'] else "❌"
                    cooldown = key_info.get('cooldown_remaining', 0)
                    cooldown_str = f" (冷却剩余: {cooldown:.0f}s)" if cooldown > 0 else ""
                    
                    click.echo(f"  {i}. {key_info['key_prefix']} {status_icon}{cooldown_str}")
                    click.echo(f"     错误次数: {key_info['error_count']}, 总请求: {key_info['total_requests']}")
        else:
            click.echo("⚠️  当前LLM客户端不支持API密钥状态查询")
            
    except Exception as e:
        click.echo(f"❌ 获取API状态失败: {e}", err=True)


def _format_json_output(result: AgentState) -> str:
    """格式化JSON输出"""
    # 安全地获取策略值
    strategy_value = None
    if hasattr(result, 'strategy') and result.strategy:
        strategy_value = result.strategy.value if hasattr(result.strategy, 'value') else str(result.strategy)
    
    output_data = {
        "strategy": strategy_value,
        "results_count": len(getattr(result, 'final_results', [])),
        "results": [
            {
                "source_table": r.source_table,
                "target_table": r.target_table,
                "score": r.score,
                "matched_columns": [
                    {
                        "source": m.source_column,
                        "target": m.target_column,
                        "confidence": m.confidence,
                        "match_type": m.match_type
                    }
                    for m in r.matched_columns
                ]
            }
            for r in getattr(result, 'final_results', [])
        ],
        "final_report": getattr(result, 'final_report', ''),
        "errors": getattr(result, 'error_messages', [])
    }
    
    return json.dumps(output_data, ensure_ascii=False, indent=2)


def _format_markdown_output(result: AgentState) -> str:
    """格式化Markdown输出"""
    lines = ["# 数据发现结果\n"]
    
    if result.strategy:
        lines.append(f"**策略**: {result.strategy.value}\n")
    
    if result.final_report:
        lines.append("## 分析报告\n")
        lines.append(result.final_report + "\n")
    
    if result.final_results:
        lines.append("## 匹配结果\n")
        
        for i, match_result in enumerate(result.final_results, 1):
            lines.append(f"### {i}. {match_result.target_table}")
            lines.append(f"- **评分**: {match_result.score:.1f}")
            lines.append(f"- **匹配列数**: {len(match_result.matched_columns)}")
            
            if match_result.matched_columns:
                lines.append("- **匹配详情**:")
                for match in match_result.matched_columns[:5]:  # 只显示前5个
                    lines.append(f"  - {match.source_column} → {match.target_column} "
                               f"(置信度: {match.confidence:.3f})")
            lines.append("")
    
    if result.error_messages:
        lines.append("## 错误信息\n")
        for error in result.error_messages:
            lines.append(f"- {error}")
        lines.append("")
    
    return "\n".join(lines)


def _format_table_output(result: AgentState) -> str:
    """格式化表格输出"""
    if not result.final_results:
        return "未找到匹配结果"
    
    # 简单的表格格式
    lines = ["排名 | 目标表 | 评分 | 匹配列数"]
    lines.append("----|------|------|--------")
    
    for i, match_result in enumerate(result.final_results, 1):
        lines.append(f"{i:2d}   | {match_result.target_table:20} | "
                    f"{match_result.score:5.1f} | {len(match_result.matched_columns):6d}")
    
    return "\n".join(lines)


if __name__ == '__main__':
    cli()