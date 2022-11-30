# BTP
Effective communication has become somewhat of a difficulty in today's multicultural societies, which are made up of people from many racial, regional, language, and ethnic backgrounds. Communication and information sharing between people of different languages and cultures depend heavily on translation. But the demand for qualified translators is even greater when it comes to a sector as significant as healthcare. because it affects people's lives! Although human translators can be used to assist, this is not a long-term fix for the issue. Machine translation is therefore used. The datasets and the way the machine is trained using the dataset are the two factors that affect machine translation. The lack of suitable datasets for the majority of the languages has created various problems. Such kind of languages are stated as low resource languages. In this project, neural machine translation model that is, transformer model is used to translate medical reports on chest x-ray into a low resource language i.e. Telugu. The medical reports are taken from mimic-cxr dataset. The model is trained using an English-Telugu parallel corpus which contains a normal English-Telugu parallel corpus combined with medical reports and their translated counterparts. Then the trained model is used to translate medical reports. The bleu and the meteor scores for this model are 37.97 and 0.79 respectively.
