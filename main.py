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

        global llm

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
