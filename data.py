from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

print(len(news.data))
print(len(news.target_names))

print(news.target_names)

for text, num_label in zip(news.data[:10], news.target[:10]):
    print('[%s]:\t\t "%s ..."' % (news.target_names[num_label], text[:100].split('\n')[0]))
    