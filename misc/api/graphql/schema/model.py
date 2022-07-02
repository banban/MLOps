from typing import Any

from ariadne import ObjectType
from data import Model, ModelPayload, all_models, all_models_by_version, get_model, update_model, get_author, get_category
from graphql import GraphQLResolveInfo

from schema.types import mutation, query

MODEL_TYPEDEF = """
    type Model {
        id: ID!
        title: String!
        content: String!
        author_id: Int!
        category_id: Int!
        author: Author!
        category: Category!
        version: String!
    }

    input ModelPayload {
        title: String
        content: String
        version: String
        author_id: Int
        category_id: Int
    }

    type Mutation {
        update_model(id: ID!, payload: ModelPayload!): Model!
    }
"""

model_query = ObjectType("Model")


@query.field("models")
def resolve_models(_, info: GraphQLResolveInfo) -> list[Model]:
    return all_models()

@query.field("models_by_version")
def resolve_models_with_version(_, info: GraphQLResolveInfo, version: str) -> list[Model]:
    return all_models_by_version(version)

@query.field("model")
def resolve_model(_, info: GraphQLResolveInfo, id: str) -> Model:
    return get_model(int(id))


@mutation.field("update_model")
def resolve_update_model(
    _, info: GraphQLResolveInfo, id: str, payload: ModelPayload
) -> Model:
    return update_model(int(id), payload)


@model_query.field("author")
def resolve_model_author(model: dict[str, Any], info: GraphQLResolveInfo):
    print(model)
    return get_author(model["author_id"])

@model_query.field("category")
def resolve_model_category(model: dict[str, Any], info: GraphQLResolveInfo):
    print(model)
    return get_category(model["category_id"])
