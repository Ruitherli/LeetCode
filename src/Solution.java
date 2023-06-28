import java.util.*;
import java.util.stream.IntStream;

public class Solution {

     //Definition for singly-linked list.
     public class ListNode {
         int val;
         ListNode next;
         ListNode() {}
         ListNode(int val) { this.val = val; }
         ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     }

    //1. Two Sum
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        int[] result = new int [2];

        for(int i = 0; i<nums.length; i++){
            //Checks if this value can be calculated to the target value
            if(map.containsKey(target-nums[i])){
                result[1] = i;
                result[0] = map.get(target-nums[i]);
                return result;
            }
            //Store the new value to a hashmap alongside its index
            map.put(nums[i],i);
        }
        return result;
    }

    //2. Add Two Numbers
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // Create a dummy head for the new linked list.
        ListNode dummyHead = new ListNode(0);

        // Initialize two pointers for the two input lists and the current pointer for the new list.
        ListNode p = l1, q = l2, curr = dummyHead;

        // Initialize carry to handle situations where sum is 10 or more.
        int carry = 0;

        // Loop through the lists as long as there is a node in at least one of them.
        while (p != null || q != null) {
            // Get the current values, if a list has no more elements, consider its value as 0.
            int x = (p != null) ? p.val : 0; // if true p.val else 0
            int y = (q != null) ? q.val : 0; // if true q.val else 0

            // Calculate the sum and carry.
            int sum = carry + x + y;
            carry = sum / 10;

            // Add the sum's unit place as a new node to our result list.
            curr.next = new ListNode(sum % 10);
            curr = curr.next;

            // Move to next elements in the lists.
            if (p != null) p = p.next;
            if (q != null) q = q.next;
        }

        // If there is a leftover carry, add it as an extra node to our list.
        if (carry > 0) {
            curr.next = new ListNode(carry);
        }

        // Return the next of dummyHead, which is the head of our result list.
        return dummyHead.next;
    }

    //3. Longest Substring Without Repeating Characters
    public int lengthOfLongestSubstring(String s) {
        String tmp = "";
        int maxLength = 0;

        for (int i = 0; i < s.length(); i++){
            if(tmp.indexOf(s.charAt(i)) > -1){
                tmp = tmp.substring(tmp.indexOf(s.charAt(i)) + 1);
            }
            tmp += s.charAt(i);
            maxLength = Math.max(maxLength, tmp.length());
        }

        return maxLength;
    }

    //5. Longest Palindromic Substring
    private static String expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return s.substring(L + 1, R);
    }

    public static String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        String longest = "";
        for (int i = 0; i < s.length(); i++) {
            String temp = expandAroundCenter(s, i, i);
            if (temp.length() > longest.length()) {
                longest = temp;
            }
            temp = expandAroundCenter(s, i, i + 1);
            if (temp.length() > longest.length()) {
                longest = temp;
            }
        }
        return longest;
    }

    //9. Palindrome Number
    public boolean isPalindrome(int x) {
        String str = String.valueOf(x);
        StringBuilder reverse = new StringBuilder();

        reverse.append(str);
        reverse.reverse();

        return str.equals(String.valueOf(reverse));
    }

    //11. Container With Most Water
    public int maxArea(int[] height) {
        int size = height.length;
        int maxArea = 1;

        for (int i = 0; i<size; i++){
            int heightOne = height[i];
            for(int j = 1; j<size; j++){
                int heightTwo = height[j];
                int temp = 0;
                if(heightOne > heightTwo){
                    temp = heightTwo * (j-i);
                } else{
                    temp = heightOne * (j-i);
                }
                if (temp > maxArea){
                    maxArea = temp;
                }
            }
        }
        return maxArea;
    }
    public int maxArea2(int[] height) {
        int maxArea = 0;
        int left = 0;
        int right = height.length - 1;

        while (left < right) {
            int h = Math.min(height[left], height[right]);
            maxArea = Math.max(maxArea, h * (right - left));

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }


    //13. Roman to Integer
    public static int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);

        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (i > 0 && map.get(s.charAt(i)) > map.get(s.charAt(i - 1))) {
                // subtract twice the value of the smaller numeral, since we added it once in the previous iteration
                result += map.get(s.charAt(i)) - 2 * map.get(s.charAt(i - 1));
            } else {
                result += map.get(s.charAt(i));
            }
        }

        return result;
    }
    //14. Longest Common Prefix
    public String longestCommonPrefix(String[] strs) {
         //Base case
        if (strs == null || strs.length == 0) {
            return "";
        }

        //Loop through each string
        for (int i = 0; i < strs[0].length() ; i++){
            char c = strs[0].charAt(i);//Get the char of the first string
            //Loops through the other strings to check if the char matches the one of the first string
            for (int j = 1; j < strs.length; j ++) {
                if ( i == strs[j].length() || strs[j].charAt(i) != c)
                    return strs[0].substring(0, i);
            }
        }
        return strs[0];
    }

    //15. 3Sum
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();

        for (int i = 0; i < nums.length - 2; i++) {
            // Skip duplicate numbers
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int low = i + 1;
            int high = nums.length - 1;

            while (low < high) {
                int sum = nums[i] + nums[low] + nums[high];

                if (sum < 0) {
                    low++;
                } else if (sum > 0) {
                    high--;
                } else {
                    // Add the triplet to the result
                    result.add(Arrays.asList(nums[i], nums[low], nums[high]));

                    // Skip duplicate numbers
                    while (low < high && nums[low] == nums[low + 1]) {
                        low++;
                    }
                    while (low < high && nums[high] == nums[high - 1]) {
                        high--;
                    }

                    // Move the pointers inward
                    low++;
                    high--;
                }
            }
        }

        return result;
    }

    //17. Letter Combinations of a Phone Number
    String[] mapping = {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    public List<String> letterCombinations(String digits) {
         List<String> result = new ArrayList<>();
        if (digits == null || digits.length() == 0) return result;
        letterCombinationsRecursive(result,digits,"",0);
        return result;
    }
    private void letterCombinationsRecursive(List<String> result, String digits, String current, int index) {
        if (index == digits.length()) {
            result.add(current);
            return;
        }

        String letters = mapping[digits.charAt(index) - '0'];
        for (int i = 0; i < letters.length(); i++) {
            letterCombinationsRecursive(result, digits, current + letters.charAt(i), index + 1);
        }
    }

    //20. Valid Parentheses
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<Character>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else if (c == ')' && !stack.isEmpty() && stack.peek() == '(') {
                stack.pop();
            } else if (c == '}' && !stack.isEmpty() && stack.peek() == '{') {
                stack.pop();
            } else if (c == ']' && !stack.isEmpty() && stack.peek() == '[') {
                stack.pop();
            } else {
                return false;
            }
        }
        return stack.isEmpty();
    }

    //21. Merge Two Sorted Lists
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        // Create a dummy head for the new linked list.
        ListNode dummyHead = new ListNode(0);

        // Initialize two pointers for the two input lists and the current pointer for the new list.
        ListNode p = list1, q = list2, curr = dummyHead;

        // Loop through the lists as long as there is a node in at least one of them.
        while (p != null && q != null) {

            if (p.val<=q.val){
                curr.next = p;

                p = p.next;
            } else {
                curr.next = q;
                q = q.next;
            }
            curr = curr.next;
        }
        // If one of the lists is empty, append the other list to the end of the new list.
        if (p != null) {
            curr.next = p;
        } else if (q != null) {
            curr.next = q;
        }
        // Return the next of dummyHead, which is the head of our result list.
        return dummyHead.next;
    }

    //26. Remove Duplicates from Sorted Array
    public static int removeDuplicates(int[] nums) {
        if(nums.length == 0) return 0;
        int i = 0; // pointer to track unique elements
        for(int j = 1; j < nums.length; j++) { // pointer to scan through the array
            if(nums[j] != nums[i]) { // if next element is not same as current
                i++; // move the unique element pointer
                nums[i] = nums[j]; // copy the next unique element to the new position
            }
        }
        return i + 1; // i is zero-indexed, so add 1 to get count of unique elements
    }

    //35. Search Insert Position
    public int searchInsert(int[] nums, int target) {
        int size = nums.length;

        for (int i = 0; i<size; i++){
            int x = nums[i];
            if (x >= target){
                return i;
            } else if (i == size-1 && target > x){
                return i+1;
            }
        }

        return 0;
    }
    //36. Valid Sudoku
    public boolean isValidSudoku(char[][] board) {
        boolean[][] rows = new boolean[9][9];
        boolean[][] cols = new boolean[9][9];
        boolean[][] boxes = new boolean[9][9];

        for (int r = 0; r < 9; r++) {
            for (int c = 0; c < 9; c++) {
                if (board[r][c] != '.') {
                    int value = board[r][c] - '1';
                    int boxIndex = r / 3 * 3 + c / 3;  // This calculates the index of the 3x3 box

                    // Check if this value already exists
                    if (rows[r][value] || cols[c][value] || boxes[boxIndex][value]) {
                        return false;
                    }

                    // Mark as existing
                    rows[r][value] = true;
                    cols[c][value] = true;
                    boxes[boxIndex][value] = true;
                }
            }
        }

        return true;
    }


    //49. Group Anagrams
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0){
            return Arrays.asList(Arrays.asList(""));
        }

        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] ca = s.toCharArray();
            Arrays.sort(ca);
            String key = String.valueOf(ca);
            if (!map.containsKey(key)){
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(s);
        }
        return new ArrayList<>(map.values());
    }

    //128. Longest Consecutive Sequence
    public int longestConsecutive(int[] nums) {
        int size = nums.length;

        if(size == 0){
            return 0;
        }
        int length = 1;
        Arrays.sort(nums);

        for (int i = 0; i<size-1; i++){
            if (nums[i+1]==(nums[i]+1)){
                length++;
            } else if (nums[i]==nums[i+1]){
                //Do nothing
            } else {
                break;
            }
        }
        return length;
    }

    //217. Contains Duplicate
    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        int size = nums.length;

        for (int i = 0; i<size-1; i++){
            if (nums[i] == nums[i+1]){
                return true;
            }
        }
        return false;
    }

    //238. Product of Array Except Self
    public int[] productExceptSelf(int[] nums) {
        int size = nums.length;
        int[] result = new int[size];

        int prefix = 1;
        result[0] = 1;
        for (int i = 0; i < size - 1; i++) {
            result[i+1] = prefix * nums[i];
            prefix = prefix * nums[i];
        }

        int postfix = 1;
        for (int j = size-1; j != -1; j--){
            result[j] = postfix * result[j];
            postfix = postfix * nums[j];
        }
        return result;
    }

    //242. Valid Anagram
    public boolean isAnagram(String s, String t) {
        char[] charsX = s.toCharArray();
        char[] charsY = t.toCharArray();

        Arrays.sort(charsX);
        Arrays.sort(charsY);

        String sortedX = new String (charsX);
        String sortedY = new String (charsY);

        int sizeX = s.length();
        int sizeY = t.length();

        if(sizeY != sizeX) {return false;}

        for(int i = 0; i<sizeX; i++){
            if (sortedX.charAt(i) != sortedY.charAt(i)){
                return false;
            }
        }
        return true;
    }

    //347. Top K Frequent Elements
    public int[] topKFrequent(int[] nums, int k) {
        // Step 1: Count the frequency of each number using a HashMap
        HashMap<Integer, Integer> count = new HashMap<>();
        for(int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        // Step 2: Create a priority queue that keeps the k most frequent elements.
        // The frequency is the priority. If two elements have the same frequency,
        // the one with the lower value is considered higher priority.
        PriorityQueue<Integer> heap = new PriorityQueue<>(
                (n1, n2) -> count.get(n1).equals(count.get(n2)) ?
                        n2 - n1 : count.get(n1) - count.get(n2)
        );

        // Step 3: Build the heap
        for(int num : count.keySet()) {
            heap.add(num);
            if(heap.size() > k) heap.poll();
        }

        // Step 4: Build the output array
        int[] top = new int[k];
        for(int i = k - 1; i >= 0; --i) {
            top[i] = heap.poll();
        }

        return top;
    }

    //744. Find Smallest Letter Greater Than Target
    public char nextGreatestLetter(char[] letters, char target) {
        // Initialize result to be the maximum character value
        char result = Character.MAX_VALUE;
        // Initialize smallest to be the maximum character value
        char smallest = Character.MAX_VALUE;

        // Iterate over each letter in the input array
        for (char letter : letters) {
            // If the current letter is greater than the target
            // and less than the current result, update result
            if (letter > target && letter < result) {
                result = letter;
            }

            // If the current letter is less than the smallest seen so far,
            // update smallest
            if (letter < smallest) {
                smallest = letter;
            }
        }

        // If result is still the maximum character value, this means no letter
        // greater than the target was found, so return the smallest letter.
        // Otherwise, return the smallest letter greater than the target.
        if (result != Character.MAX_VALUE) {
            return result;
        } else {
            return smallest;
        }
    }


    //1318. Minimum Flips to Make a OR b Equal to c
    public int minFlips(int a, int b, int c) {
        int flips = 0;
        while (a > 0 || b > 0 || c > 0) {
            //Get the AND results of the rightmost bit
            int bitA = a & 1;
            int bitB = b & 1;
            int bitC = c & 1;

            //If C == 0, three conditions can be met
            //1. A == 0, B == 0 , no flips
            //2. A == 1, B == 0, one flip
            //3. A == 1, B == 1, two flips
            //To calculate the flips needed, the sum of A and B can be used
            if (bitC == 0) {
                flips += (bitA + bitB);
            } else {
                //If C == 1, three conditions can be met
                //1. A == 0, B == 0 , one flip
                //2. A == 1, B == 0, no flips
                //3. A == 1, B == 1, no flips
                //Only case 1 need to be considered
                if (bitA == 0 && bitB == 0) {
                    flips += 1;
                }
            }

            //Shift the bits to the right by 1 time so the last bit now is previously the last second bit
            //Example, 11001 --> 01100 --> 00110 --> 00011 --> 00001 --> 00000
            a >>= 1;
            b >>= 1;
            c >>= 1;
        }

        return flips;
    }

    //1351. Count Negative Numbers in a Sorted Matrix
    public int countNegatives(int[][] grid) {
        int counter = 0;

        for (int i = 0; i<grid.length; i++){
            int[] row = grid[i];
            Arrays.sort(row);
            for (int j = 0; j<row.length; j++){
                if (row[j]>=0){
                    break;
                } else{
                    counter++;
                }
            }
        }
        return counter;
    }
    //1502. Can Make Arithmetic Progression From Sequence
    public boolean canMakeArithmeticProgression(int[] arr) {
        Arrays.sort(arr);
        for(int i = 0; i<arr.length-2 ; i++){
            if (arr[i+1]-arr[i] != arr[i+2]-arr[i+1]){
                return false;
            }
        }
        return true;
    }
    //1802. Maximum Value at a Given Index in a Bounded Array
    public int maxValue(int n, int index, int maxSum) {
        int low = 1, high = maxSum; // Initialize the binary search range to [1, maxSum]

        // Perform the binary search
        while (low < high) {
            // Calculate the mid value. Note that we add 1 to (high - low) / 2 to ensure that mid is closer to high when low and high are both even.
            long mid = (high - low + 1) / 2 + low;

            // Calculate the sum of the left subarray
            long sumLeft = ((mid - index) + (mid - 1)) * index / 2;
            if (mid <= index) {
                // If mid is less than or equal to index, the left subarray is a complete arithmetic sequence, so we calculate the sum differently.
                sumLeft = (mid - 1) * mid / 2 + index - mid + 1;
            }
            sumLeft = Math.max(sumLeft, index);  // The sum of the left subarray cannot be less than index.

            // Calculate the sum of the right subarray
            long sumRight = ((mid - 1) + (mid - (n - 1 - index))) * (n - index - 1) / 2;
            if (mid <= n - 1 - index) {
                // If mid is less than or equal to n - 1 - index, the right subarray is a complete arithmetic sequence, so we calculate the sum differently.
                sumRight = (mid - 1) * mid / 2 + (n - 1 - index - mid) + 1;
            }
            sumRight = Math.max(sumRight, n - index - 1);  // The sum of the right subarray cannot be less than n - index - 1.

            // Calculate the total sum
            long sum = sumLeft + sumRight + mid;

            // If the total sum is greater than maxSum, decrease high. Otherwise, increase low.
            if (sum > maxSum)
                high = (int) mid - 1;
            else
                low = (int) mid;
        }

        // After the binary search, low is the maximum value nums[index] can be.
        return low;
    }
}
