"""
SQLAlchemy declarative base.

All ORM models inherit from `Base`. Keeping it in a tiny file avoids
circular imports when models reference each other.


SQLAlchemy 声明式基类。
所有 ORM 模型都继承自 `Base`。将其放在一个很小的文件中，可以避免模型相互引用时出现循环导入的问题
"""
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """所有ORM模型的声明式基类。"""