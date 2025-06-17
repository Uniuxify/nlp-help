import inspect
from pandas.io import clipboard
import numpy as np


def func(name):
    clipboard.copy(dict_[name])

llm = None

def red_button(text):
    try:
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        from langchain.prompts import ChatPromptTemplate, PromptTemplate
        # from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnablePassthrough, RunnableParallel, \
        #     RunnableBranch
        # from langchain.schema.output_parser import StrOutputParser
        # from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

        from langchain_openai import ChatOpenAI

        if not llm:
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1/",
                model="deepseek/deepseek-chat-v3-0324:free",
                api_key="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",
            )

        system_prompt = "Ты — эксперт и хороший программист " \
                        "В ответе на запросы просто пиши код без комментариев и дополнительных пояснений (если не сказано обратного)"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]

        response = llm.invoke(messages).content
        clipboard.copy(response)
    except Exception as e:
        clipboard.copy(f'!failed: {e}')


dict_ = {
    'test_func': '''
def test_func():
    print('success!')
'''
}
