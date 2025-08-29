from typing import List, Dict, Any, Optional
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from src.core.workflow import create_workflow, discover_data  # Old workflow removed
from src.core.multi_agent_workflow import create_workflow, DataLakeDiscoveryWorkflow
# Note: discover_data function no longer exists, needs refactoring if API is used
from src.core.models import AgentState, TableInfo, ColumnInfo
from src.config.settings import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="基于大语言模型的数据湖模式匹配与数据发现系统"
)

# 全局工作流实例
workflow = create_workflow()


# API数据模型
class DiscoveryRequest(BaseModel):
    """数据发现请求模型"""
    user_query: str
    query_tables: Optional[List[Dict[str, Any]]] = None
    query_columns: Optional[List[Dict[str, Any]]] = None


class DiscoveryResponse(BaseModel):
    """数据发现响应模型"""
    success: bool
    strategy: Optional[str] = None
    results: List[Dict[str, Any]] = []
    final_report: str = ""
    processing_log: List[str] = []
    error_messages: List[str] = []


class TableIndexRequest(BaseModel):
    """表索引请求模型"""
    tables: List[Dict[str, Any]]


class ColumnIndexRequest(BaseModel):
    """列索引请求模型"""
    columns: List[Dict[str, Any]]


# API端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "数据湖多智能体系统",
        "version": settings.version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "workflow_info": workflow.get_workflow_info()
    }


@app.post("/discover", response_model=DiscoveryResponse)
async def discover_tables_and_columns(request: DiscoveryRequest):
    """
    执行数据发现
    
    支持两种模式：
    1. 表头匹配（Schema Matching）：提供query_columns
    2. 数据实例匹配（Data Instance Matching）：提供query_tables
    """
    try:
        logger.info(f"收到数据发现请求: {request.user_query}")
        
        # 验证请求
        if not request.query_tables and not request.query_columns:
            raise HTTPException(
                status_code=400,
                detail="必须提供 query_tables 或 query_columns 中的至少一个"
            )
        
        # 转换数据格式
        parsed_tables = []
        if request.query_tables:
            for table_data in request.query_tables:
                try:
                    table_info = TableInfo(**table_data)
                    parsed_tables.append(table_info)
                except Exception as e:
                    logger.error(f"解析表数据失败: {e}")
                    raise HTTPException(status_code=400, detail=f"表数据格式错误: {e}")
        
        parsed_columns = []
        if request.query_columns:
            for col_data in request.query_columns:
                try:
                    col_info = ColumnInfo(**col_data)
                    parsed_columns.append(col_info)
                except Exception as e:
                    logger.error(f"解析列数据失败: {e}")
                    raise HTTPException(status_code=400, detail=f"列数据格式错误: {e}")
        
        # 创建初始状态
        initial_state = AgentState(
            user_query=request.user_query,
            query_tables=parsed_tables,
            query_columns=parsed_columns
        )
        
        # 执行工作流
        final_state = await workflow.run(initial_state)
        
        # 构建响应
        results = []
        for match_result in final_state.final_results:
            result_dict = {
                "source_table": match_result.source_table,
                "target_table": match_result.target_table,
                "score": match_result.score,
                "matched_columns_count": len(match_result.matched_columns),
                "matched_columns": [
                    {
                        "source": m.source_column,
                        "target": m.target_column,
                        "confidence": m.confidence,
                        "match_type": m.match_type,
                        "reason": m.reason
                    }
                    for m in match_result.matched_columns
                ],
                "evidence": match_result.evidence
            }
            results.append(result_dict)
        
        response = DiscoveryResponse(
            success=len(final_state.error_messages) == 0,
            strategy=final_state.strategy.value if final_state.strategy else None,
            results=results,
            final_report=final_state.final_report,
            processing_log=final_state.processing_log,
            error_messages=final_state.error_messages
        )
        
        logger.info(f"数据发现完成，返回 {len(results)} 个结果")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据发现处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@app.post("/discover/stream")
async def discover_with_streaming(request: DiscoveryRequest):
    """
    流式数据发现（Server-Sent Events）
    """
    try:
        # 解析输入数据（复用上面的逻辑）
        parsed_tables = []
        if request.query_tables:
            for table_data in request.query_tables:
                table_info = TableInfo(**table_data)
                parsed_tables.append(table_info)
        
        parsed_columns = []
        if request.query_columns:
            for col_data in request.query_columns:
                col_info = ColumnInfo(**col_data)
                parsed_columns.append(col_info)
        
        initial_state = AgentState(
            user_query=request.user_query,
            query_tables=parsed_tables,
            query_columns=parsed_columns
        )
        
        # 流式执行
        from fastapi.responses import StreamingResponse
        import json
        
        async def event_generator():
            async for state in workflow.run_with_streaming(initial_state):
                # 构建流式响应数据
                event_data = {
                    "current_step": state.current_step,
                    "progress": len(state.processing_log),
                    "latest_log": state.processing_log[-1] if state.processing_log else "",
                    "partial_results_count": len(state.final_results),
                    "errors": state.error_messages
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"流式数据发现失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/tables")
async def index_tables(request: TableIndexRequest):
    """
    为表建立索引
    """
    try:
        logger.info(f"开始为 {len(request.tables)} 个表建立索引")
        
        # 解析表数据
        table_infos = []
        for table_data in request.tables:
            table_info = TableInfo(**table_data)
            table_infos.append(table_info)
        
        # 建立表索引
        await workflow.table_discovery.initialize_table_index(table_infos)
        
        logger.info("表索引建立完成")
        return {"message": f"成功为 {len(table_infos)} 个表建立索引"}
        
    except Exception as e:
        logger.error(f"表索引建立失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/columns")
async def index_columns(request: ColumnIndexRequest):
    """
    为列建立索引
    """
    try:
        logger.info(f"开始为 {len(request.columns)} 个列建立索引")
        
        # 解析列数据
        column_infos = []
        for col_data in request.columns:
            col_info = ColumnInfo(**col_data)
            column_infos.append(col_info)
        
        # 建立列索引
        await workflow.column_discovery.initialize_indices(column_infos)
        
        logger.info("列索引建立完成")
        return {"message": f"成功为 {len(column_infos)} 个列建立索引"}
        
    except Exception as e:
        logger.error(f"列索引建立失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """
    获取系统统计信息
    """
    try:
        # TODO: 实现统计信息收集
        stats = {
            "indexed_tables": 0,  # 从向量搜索引擎获取
            "indexed_columns": 0,  # 从搜索引擎获取
            "total_queries": 0,   # 从日志或计数器获取
            "avg_response_time": 0  # 从性能监控获取
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index")
async def clear_indices():
    """
    清空所有索引
    """
    try:
        # TODO: 实现索引清空逻辑
        logger.info("清空所有索引")
        return {"message": "所有索引已清空"}
        
    except Exception as e:
        logger.error(f"清空索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # 创建必要的目录
    settings.create_directories()
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.logging.level.lower()
    )