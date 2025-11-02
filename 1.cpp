#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>
using namespace std;

//base 62 encoding
string base62encode(unsigned long long num){
	const string base62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	if(num == 0) return "0";

	string result;
	while(num>0){
		result += base62[num % 62];
		num /= 62;
	}
	reverse(result.begin(), result.end());
	return result;
}

//base62 decoding
string base62decode()

// Generating short code
void hashFunction(string url){
	hash<string> stringHash;
	unordered_map <string , string> mp;
	string value = base62encode(stringHash(url));
	mp[url] = value;
	cout << value << endl;
	cout << stringHash(url) << endl;
}

int main(){
	hashFunction("youtube.com");
}
