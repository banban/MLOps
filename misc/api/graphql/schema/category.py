from data import Category, all_categories, get_category
from graphql.type.definition import GraphQLResolveInfo

from schema.types import query

CATEGORY_TYPEDEF = """
    type Category {
        id: ID!
        name: String!
    }
"""


@query.field("categories")
def resolve_categories(_, info: GraphQLResolveInfo) -> list[Category]:
    return all_categories()


@query.field("category")
def resolve_category(_, info: GraphQLResolveInfo, id: str) -> Category:
    return get_category(int(id))
