## Local Image Search

1. **Caching of Image Descriptors:** Gets all images in the "photos" directory, computes their ORB feature descriptors, and saves them to a file called 'descriptor.pkl':

```python
add("photos", 'descriptor.pkl')
```

2. **Searching for Similar Images:** Gets the descriptor of the 'demo.jpg' image, compares it to the descriptors in the 'descriptor.pkl' file, and returns the filenames of the top 5 images in the directory that are most similar to the input image:

```python
search('demo.jpg', 'descriptor.pkl', 5)
```
This will return a list of the filenames of the top 5 images in the directory that are similar to the input image:

```python
['774909.jpg', '220453.jpg', '3866555.jpg', '1239291.jpg', '1222271.jpg']
```
