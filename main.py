from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Charles H. Robbins (born 1965 or 1966) is an American businessman. He is the chairman and CEO of Cisco Systems.[1]

Early life
Robbins was born in Grayson, Georgia, and educated at Rocky Mount High School in Rocky Mount, North Carolina.[2] In 1987, he earned a Bachelor of Mathematics degree from the University of North Carolina at Chapel Hill.[1]

Career
Robbins began his career as an application developer for North Carolina National Bank (now part of Bank of America). He joined Wellfleet Communications followed by a brief tenure at Ascend Communications before joining Cisco in 1997.[2]

Robbins filled various posts at Cisco, including senior vice president of the Americas and senior vice president of Worldwide Field Operations, a role in which he led Cisco's Worldwide Sales and Partner Organizations and built out Cisco's partnership program.[3][4]

In May 2015, Cisco announced that CEO and chairman John Chambers would step down as CEO in July 2015 while remaining as chairman. Robbins, then a senior vice president, was named as his successor.[5] Mentored by Chambers, Robbins was unanimously voted in as the company's new chief executive, becoming CEO of Cisco Systems in July 2015.[6][5][7]

As CEO, Robbins became noted for accelerating the pace of Cisco's modern growth,[2][8] while disrupting outdated working modes,[9] promoting employee trust based in transparency of policy and process,[10] and humanitarian policies and workplace diversity.[11][12][13]

Robbins has advocated for corporate social responsibility.[14] However, in 2018, he said that talk of corporate social responsibility is becoming obsolete and just expected of "corporate icons."[15]

In 2018, as the GDPR came into effect, Robbins called for more regulation and for the tech industry to help educate regulators. In February 2019, Robbins promoted the need for comprehensive global privacy legislation, asserting privacy as “a fundamental human right."[16]

In 2019, Robbins advocated against a 15% increase on tariffs for Chinese goods.[17][18]

In 2023, Robbins's total compensation at Cisco was $31.8 million, up 37% from the previous year and representing a CEO-to-median worker pay ratio of 267-to-1.[19] For 2024, Robbins's total compensation from Cisco was $38.2 million.[20] In 2024, Robbins was named one of the top 10 highest-paid CEOs in the world.[21]"""

    summary_template = """ 
    given the following information {information} about a peron I want you to create:
    1. A short summary of the person
    2. A list of 2 key facts about the person"""

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":   
    main()
