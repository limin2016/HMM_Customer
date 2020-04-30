from CustomerHMM import CustomerHMM

# You can replace 'hmm_customer_1586733275338.txt' with other input files
a = CustomerHMM()
a.get_observation_list('hmm_customer_1586733275338.txt')
a.get_hidden_states()
