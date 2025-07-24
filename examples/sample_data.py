#!/usr/bin/env python3
"""
示例数据生成器
"""
import json
from pathlib import Path


def create_sample_tables():
    """创建示例表数据"""
    tables = [
        {
            "table_name": "customers",
            "columns": [
                {
                    "table_name": "customers",
                    "column_name": "customer_id",
                    "data_type": "integer",
                    "sample_values": [1, 2, 3, 100, 201],
                    "unique_count": 1000,
                    "null_count": 0
                },
                {
                    "table_name": "customers",
                    "column_name": "name",
                    "data_type": "varchar",
                    "sample_values": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"],
                    "unique_count": 995,
                    "null_count": 5
                },
                {
                    "table_name": "customers",
                    "column_name": "email",
                    "data_type": "varchar",
                    "sample_values": ["john@example.com", "jane@test.com", "bob@sample.org"],
                    "unique_count": 1000,
                    "null_count": 0
                },
                {
                    "table_name": "customers",
                    "column_name": "age",
                    "data_type": "integer",
                    "sample_values": [25, 30, 45, 22, 67],
                    "unique_count": 50,
                    "null_count": 10
                }
            ],
            "row_count": 1000,
            "description": "客户基本信息表"
        },
        {
            "table_name": "orders",
            "columns": [
                {
                    "table_name": "orders",
                    "column_name": "order_id",
                    "data_type": "integer",
                    "sample_values": [1001, 1002, 1003, 1004],
                    "unique_count": 5000,
                    "null_count": 0
                },
                {
                    "table_name": "orders",
                    "column_name": "customer_id",
                    "data_type": "integer",
                    "sample_values": [1, 2, 3, 100, 201],
                    "unique_count": 800,
                    "null_count": 0
                },
                {
                    "table_name": "orders",
                    "column_name": "product_name",
                    "data_type": "varchar",
                    "sample_values": ["Laptop", "Mouse", "Keyboard", "Monitor"],
                    "unique_count": 200,
                    "null_count": 0
                },
                {
                    "table_name": "orders",
                    "column_name": "amount",
                    "data_type": "decimal",
                    "sample_values": [999.99, 29.99, 79.99, 299.99],
                    "unique_count": 1000,
                    "null_count": 0
                }
            ],
            "row_count": 5000,
            "description": "订单信息表"
        },
        {
            "table_name": "users",
            "columns": [
                {
                    "table_name": "users",
                    "column_name": "user_id",
                    "data_type": "integer",
                    "sample_values": [1, 2, 3, 4, 5],
                    "unique_count": 1200,
                    "null_count": 0
                },
                {
                    "table_name": "users",
                    "column_name": "username",
                    "data_type": "varchar",
                    "sample_values": ["john_doe", "jane_smith", "bob_j", "alice_b"],
                    "unique_count": 1200,
                    "null_count": 0
                },
                {
                    "table_name": "users",
                    "column_name": "email_address",
                    "data_type": "varchar",
                    "sample_values": ["john@example.com", "jane@test.com", "bob@demo.net"],
                    "unique_count": 1200,
                    "null_count": 0
                },
                {
                    "table_name": "users",
                    "column_name": "registration_date",
                    "data_type": "datetime",
                    "sample_values": ["2023-01-15", "2023-02-20", "2023-03-10"],
                    "unique_count": 365,
                    "null_count": 0
                }
            ],
            "row_count": 1200,
            "description": "用户账户信息表"
        }
    ]
    
    return tables


def create_sample_columns():
    """创建示例列数据"""
    columns = [
        {
            "table_name": "query_table", 
            "column_name": "id",
            "data_type": "integer",
            "sample_values": [1, 2, 3, 4, 5],
            "unique_count": 100,
            "null_count": 0
        },
        {
            "table_name": "query_table",
            "column_name": "customer_name", 
            "data_type": "varchar",
            "sample_values": ["John Doe", "Jane Smith", "Bob Johnson"],
            "unique_count": 95,
            "null_count": 5
        },
        {
            "table_name": "query_table",
            "column_name": "email_addr",
            "data_type": "varchar", 
            "sample_values": ["john@example.com", "jane@test.com", "bob@sample.org"],
            "unique_count": 100,
            "null_count": 0
        }
    ]
    
    return columns


def main():
    """主函数"""
    # 创建示例目录
    examples_dir = Path(__file__).parent
    examples_dir.mkdir(exist_ok=True)
    
    # 生成示例表数据
    tables = create_sample_tables()
    with open(examples_dir / "sample_tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    
    # 生成示例列数据
    columns = create_sample_columns()
    with open(examples_dir / "sample_columns.json", "w", encoding="utf-8") as f:
        json.dump(columns, f, ensure_ascii=False, indent=2)
    
    print("✅ 示例数据已生成:")
    print(f"  - 表数据: {examples_dir / 'sample_tables.json'}")
    print(f"  - 列数据: {examples_dir / 'sample_columns.json'}")


if __name__ == "__main__":
    main()