import certifi
from haystack.document_stores import ElasticsearchDocumentStore, \
  OpenSearchDocumentStore
from haystack.nodes import EmbeddingRetriever

cert_where = certifi.where()
print(f"ca_certs file path: {cert_where}")

document_store_1 = ElasticsearchDocumentStore(
    embedding_dim=128,
    index="store_1",
    similarity="cosine",
    host="localhost",
    port=9200,
    scheme="http",
    username="",
    verify_certs=False,
    ca_certs=cert_where
)
retriever_1 = EmbeddingRetriever(
    document_store_1, embedding_model="yjernite/retribert-base-uncased",
    model_format="retribert")
dicts_1 = [
  {
    'content':
      """Document text one alpha beta gamma Investors can upload, request, and view documents from the company documents page.
         To view the company documents page, click the company's name or logo you would like to view on the Company Investments page or on the side bar.
         Convertible debt: Will have fields related to calculating simple or compound interest of the convertible. Some fields include:
The valuation cap sets a maximum value at which a convertible security will convert into equity in the financing round.
The conversion discount is the discount rate at which the note holder will purchase shares in the next financing round.
The interest accrual period is the length of time over which the accrued  interest is shown on the convertible details.
Upload default or template documents to attach for each convertible as they are individually drafted.
NOTE: To issue convertibles in Carta the Form of Convertible is required, the Note purchase agreement is optional.
The Vesting library supports two kinds of vesting criteria that can be created as a template to easily use when issuing new interests:

Time (or Service): Vests after an amount of time has elapsed. For example, 10,000 units vest 1/48 each month for 48 months.
Performance: Vests after an event has taken place. For example, 5,000 units vest when the enterprise value has been increased by 2x.
 
Each Vesting plan template may have any combination of Time and/or Performance criteria for a given tranche of units. 
Users with Full administrator permissions can create performance conditions and standard or custom vesting templates.
 

Performance conditions
Performance conditions are intended to be used on interests that have a time-based vesting schedule, but do not start 
vesting until the performance-based vesting condition is met. At the time of performance achievement, the time-based
 vesting that occurred in the past, will be attributed in the current period and any time-based vesting remaining for 
 future periods will accrue based on the time-schedule.
      """
  },
  {
    'content':
      """Document text two alpha beta gamma Accelerate vesting is not available for exercised grants or grants with milestone 
      vesting schedules. Milestone vesting tranches are not automatically triggered, regardless of the vesting date. 
      Because Milestones are simply text fields for vesting conditions, the Carta platform is not able to recognize that the 
      vesting terms have been met. For these cases, submit a case via Carta's Help Center. Learn more about Carta's Help Center.
      Effective date: Determines when this amendment affects as-of date reporting. Usually the approval date for the amendment.
      NOTE: When accelerating vesting to reflect the amount of options of an ongoing exercise, the effective date must be 
      prior to the exercise submission date so the update can be effective. Consult with your legal advisor before modifying 
      vesting schedules, including for situations as noted, as changes to vesting schedules often need Board approval.
Vesting acceleration date: The date on which the accelerated shares will vest.
Accelerate quantity: Amount of shares being vested by accelerating vesting.
 
As these three pieces of information are entered, the administrator will be able to preview the accelerated vesting 
schedule as the schedule will automatically calculate and create a modified version. The accelerated quantity will be pulled 
from the next consecutive tranches until the last modified tranche shows the remaining balance from the original vesting schedule.
      """
  }
]
document_store_1.delete_documents()
document_store_1.write_documents(dicts_1)
document_store_1.update_embeddings(retriever_1, batch_size=10)

documents = retriever_1.retrieve(query="Alpha Beta Gamma Delta investors 409a platform grants vesting")
for doc in documents:
  print(doc, doc.score)
