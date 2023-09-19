import org.w3c.dom.Node;

import javax.swing.tree.TreeNode;
import java.util.*;
import java.util.stream.Collectors;
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
        HashSet<Character> set = new HashSet<>();
        int maxLength = 0;
        int left = 0, right = 0;

        while (right < s.length()){
            if(!set.contains(s.charAt(right))){
                set.add(s.charAt(right));
                maxLength = Math.max(maxLength, right - left + 1);
                right++;
            } else {
                set.remove(s.charAt(left));
                left++;
            }
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

    //19. Remove Nth Node From End of List
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p = dummy, q = dummy;
        int counter = 0;

        // Move p n steps ahead
        while (counter <= n) {
            p = p.next;
            counter++;
        }

        // Move both pointers until p reaches the end
        while (p != null) {
            p = p.next;
            q = q.next;
        }

        // Remove the n-th node from the end
        q.next = q.next.next;

        // Return the new head
        return dummy.next;

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

    //22. Generate Parentheses
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateParenthesis(n, n, "", result);
        return result;
    }

    private void generateParenthesis(int open, int close, String current, List<String> result) {
        if (open == 0 && close == 0) {
            result.add(current);
            return;
        }

        if (open > 0) {
            generateParenthesis(open - 1, close, current + "(", result);
        }
        if (close > open) {
            generateParenthesis(open, close - 1, current + ")", result);
        }
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

    //28. Find the Index of the First Occurrence in a String
    public int strStr(String haystack, String needle) {
        int haystackSize = haystack.length();
        int needleSize = needle.length();

        if (needleSize > haystackSize){
            return -1;
        }

        if (needle.equals(haystack)){
            return 0;
        }

        for (int i = 0; i <= haystackSize-needleSize; i++){
            String target = haystack.substring(i,i+needleSize);
            if (target.equals(needle)){
                return i;
            }
        }

        return -1;
    }

    //33. Search in Rotated Sorted Array
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right){
            int mid = left + (right - left)/2;
            int midValue = nums[mid];

            if (midValue == target){
                return mid;
            }

            if (nums[mid] >= nums[left]){
                if (target >= nums[left] && target < midValue){
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else{
                if (target <= nums[right] && target > midValue){
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return -1;
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

    //42. Trapping Rain Water
    public int trap(int[] height) {
        int size = height.length;
        int[] volume = new int[size];
        int waterSpots = 0;

        int leftPointer = 0;
        int rightPointer = size-1;

        int maxLeft = height[leftPointer];
        int maxRight = height[rightPointer];

        while (leftPointer != rightPointer){
            if(maxLeft <= maxRight){
                leftPointer++;
                int spots = maxLeft - height[leftPointer];
                if (spots>0) {
                    waterSpots += spots;
                }
                if(height[leftPointer]>maxLeft){
                    maxLeft = height[leftPointer];
                }
            }else{
                rightPointer--;
                int spots = maxRight - height[rightPointer];
                if (spots>0) {
                    waterSpots += spots;
                }
                if(height[rightPointer]>maxRight){
                    maxRight = height[rightPointer];
                }
            }
        }

        for(Integer vol : volume){
            waterSpots = waterSpots + vol;
        }
        return waterSpots;
    }

    //46. Permutations
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums);
        return result;
    }

    private void backtrack(List<List<Integer>> result, List<Integer> currentPermutation, int[] nums) {
        // Base case: if the current permutation's length is equal to the original array's length,
        // add it to the result.
        if (currentPermutation.size() == nums.length) {
            result.add(new ArrayList<>(currentPermutation));
            return;
        }

        // Recursive step: for every number in the original list that's not yet used in the current permutation,
        // append it to the current permutation and recurse.
        for (int i = 0; i < nums.length; i++) {
            if (currentPermutation.contains(nums[i])) continue;  // skip if element already used
            currentPermutation.add(nums[i]);
            backtrack(result, currentPermutation, nums);
            currentPermutation.remove(currentPermutation.size() - 1);  // remove the last element to backtrack
        }
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

    //58. Length of Last Word
    public int lengthOfLastWord(String s) {
        int length = 0;

        // We are looking for the last word so let's go backward
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) != ' ') { // a letter is found so count
                length++;
            } else {  // it's a white space instead
                //  Did we already started to count a word ? Yes so we found the last word
                if (length > 0) return length;
            }
        }
        return length;
    }

    //70. Climbing Stairs
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    //74. Search a 2D Matrix
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int left = 0;
        int right = m * n - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int mid_val = matrix[mid / n][mid % n];

            if(mid_val == target){
                return true;
            }else if (mid_val < target){
                left = mid + 1;
            }else{
                right = mid -1;
            }
        }

        return false;
    }

    //88. Merge Sorted Array
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m-1;
        int p2 = n-1;
        int position = m+n-1;

        while (p1 >= 0 && p2 >= 0) {
            if (nums1[p1] > nums2[p2]) {
                nums1[position] = nums1[p1];
                p1--;
            } else {
                nums1[position] = nums2[p2];
                p2--;
            }
            position--;
        }

        // If there are remaining elements in nums2, copy them over
        while (p2 >= 0) {
            nums1[position] = nums2[p2];
            position--;
            p2--;
        }
    }


    //92. Reverse Linked List II
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if(head == null) return null;
        ListNode dummy = new ListNode(0); // create a dummy node to mark the head of this list
        dummy.next = head;
        ListNode pre = dummy; // make a pointer pre as a marker for the node before reversing
        for(int i = 0; i<m-1; i++) pre = pre.next;

        ListNode start = pre.next; // a pointer to the beginning of a sub-list that will be reversed
        ListNode then = start.next; // a pointer to a node that will be reversed

        // 1 - 2 -3 - 4 - 5 ; m=2; n =4 ---> pre = 1, start = 2, then = 3
        // dummy-> 1 -> 2 -> 3 -> 4 -> 5

        for(int i=0; i<n-m; i++)
        {
            start.next = then.next;
            then.next = pre.next;
            pre.next = then;
            then = start.next;
        }

        // first reversing : dummy->1 - 3 - 2 - 4 - 5; pre = 1, start = 2, then = 4
        // second reversing: dummy->1 - 4 - 3 - 2 - 5; pre = 1, start = 2, then = 5 (finish)

        return dummy.next;

    }

    //97. Interleaving String
    public boolean isInterleave(String s1, String s2, String s3) {
        int sizeOne = s1.length();
        int sizeTwo = s2.length();
        int sizeThree = s3.length();

        if (sizeOne + sizeTwo != sizeThree) {
            return false;
        }

        boolean[][] dp = new boolean[sizeOne + 1][sizeTwo + 1];
        dp[0][0] = true;

        // Initialize the first row
        for (int i = 1; i <= sizeOne; i++) {
            dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
        }

        // Initialize the first column
        for (int j = 1; j <= sizeTwo; j++) {
            dp[0][j] = dp[0][j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);
        }

        // Fill the rest of the table
        for (int i = 1; i <= sizeOne; i++) {
            for (int j = 1; j <= sizeTwo; j++) {
                if (s1.charAt(i - 1) == s3.charAt(i + j - 1)) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
                if (s2.charAt(j - 1) == s3.charAt(i + j - 1)) {
                    dp[i][j] = dp[i][j] || dp[i][j - 1];
                }
            }
        }

        return dp[sizeOne][sizeTwo];
    }

    //104. Maximum Depth of Binary Tree
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    //118. Pascal's Triangle
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();

        for (int i = 0; i < numRows; i++) {
            List<Integer> row = new ArrayList<>();
            row.add(1); // Every row starts with a '1'

            for (int j = 1; j < i; j++) { // Start from 1 because the 0th element is already added
                int left = res.get(i-1).get(j-1);  // Left value from the previous row
                int right = res.get(i-1).get(j);   // Right value from the previous row
                row.add(left + right);
            }

            if (i > 0) { // Every row except the first ends with a '1'
                row.add(1);
            }

            res.add(row);
        }
        return res;
    }

    //119. Pascal's Triangle II
    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<>();
        row.add(1);  // Initialize with the 0th row.

        for (int i = 1; i <= rowIndex; i++) {
            for (int j = row.size() - 1; j > 0; j--) {
                row.set(j, row.get(j) + row.get(j - 1));
            }
            row.add(1);  // Append 1 at the end of each iteration.
        }

        return row;
    }


    //121. Best Time to Buy and Sell Stock
    public int maxProfit(int[] prices) {
        int left = 0; //Buy at lowest
        int right = 1; //Sell at highest
        int maxProfit = 0;

        while (right < prices.length) {
            if (prices[left] < prices[right]){
                int profit = prices[right] - prices[left];
                maxProfit = Math.max(maxProfit,profit);

            } else {
                left = right;
            }
            right++;

        }
        return maxProfit;
    }

    //125. Valid Palindrome
    public boolean isPalindrome(String s) {
        StringBuilder reverse = new StringBuilder();
        String formatted = s.trim().toLowerCase().replaceAll("[^a-zA-Z0-9]", "");

        reverse.append(formatted);
        reverse.reverse();

        return String.valueOf(reverse).equals(formatted);
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

    //136. Single Number
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    //138. Copy List with Random Pointer
    class Node {
        public int val;
        public Node next;
        public Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public Node copyRandomList(Node head) {
        Map<Node, Node> oldToCopy = new HashMap<>();
        oldToCopy.put(null, null);

        Node cur = head;
        while (cur != null) {
            Node copy = new Node(cur.val);
            oldToCopy.put(cur, copy);
            cur = cur.next;
        }

        cur = head;
        while (cur != null) {
            Node copy = oldToCopy.get(cur);
            copy.next = oldToCopy.get(cur.next);
            copy.random = oldToCopy.get(cur.random);
            cur = cur.next;
        }

        return oldToCopy.get(head);
    }

    //141. Linked List Cycle
    public boolean hasCycle(ListNode head) {
        ListNode fast = head, slow = head;

        while (fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;

            if (fast == slow){
                return true;
            }
        }

        return false;
    }

    //143. Reorder List
    public ListNode findMiddle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    public ListNode reverse(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode nextNode = head.next;
            head.next = prev;
            prev = head;
            head = nextNode;
        }
        return prev;
    }
    public void merge(ListNode l1, ListNode l2) {
        while (l1 != null && l2 != null) {
            ListNode temp1 = l1.next;
            ListNode temp2 = l2.next;

            l1.next = l2;
            if (temp1 != null) {
                l2.next = temp1;
            }

            l1 = temp1;
            l2 = temp2;
        }
    }
    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return;

        // 1. Find the middle
        ListNode middle = findMiddle(head);
        ListNode secondHalf = middle.next;
        middle.next = null;  // Split the list

        // 2. Reverse the second half
        secondHalf = reverse(secondHalf);

        // 3. Merge the two halves
        merge(head, secondHalf);
    }

    //150. Evaluate Reverse Polish Notation
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();

        for(String token : tokens){
            if (token.matches("-?\\d+")){
                stack.push(Integer.parseInt(token));
            } else {
                int val2 = stack.pop();
                int val1 = stack.pop();

                switch(token) {
                    case "+":
                        stack.push(val1 + val2);
                        break;
                    case "-":
                        stack.push(val1 - val2);
                        break;
                    case "*":
                        stack.push(val1 * val2);
                        break;
                    case "/":
                        stack.push(val1 / val2);
                        break;
                }
            }
        }

        return stack.pop();
    }

    //153. Find Minimum in Rotated Sorted Array
    public int findMin(int[] nums) {
        int res = nums[0];
        int left = 0;
        int right = nums.length - 1;

        while (left <= right){
         if (nums[left] < nums[right]){
             res = Math.min(res,nums[left]);
             break;
         }
         int mid = left + (right - left)/2;
         res = Math.min(res,nums[mid]);

         if (nums[mid] >= nums[left]){
             left = mid + 1;
         } else {
             right = mid - 1;
         }
        }
        return res;
    }

    //155. Min Stack
    public class MinStack {
        private Stack<Integer> stack;
        private Stack<Integer> minStack;

        /** initialize your data structure here. */
        public MinStack() {
            stack = new Stack<>();
            minStack = new Stack<>();
        }

        public void push(int val) {
            if (minStack.isEmpty() || val <= getMin()) {
                minStack.push(val);
            }
            stack.push(val);
        }

        public void pop() {
            if (stack.peek().equals(minStack.peek())) {
                minStack.pop();
            }
            stack.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }


    //167. Two Sum II - Input Array Is Sorted
    public int[] twoSumII(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[] { left + 1, right + 1 };
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }

    //169. Majority Element
    public int majorityElement(int[] nums) {
        int count = 0;
        Integer candidate = null;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }

        // The second pass to confirm the candidate is the majority is optional
        // since the problem states a majority will always exist.
        // If not guaranteed, you'd iterate over nums again and ensure candidate appears more than n/2 times.

        return candidate;
    }


    //206. Reverse Linked List
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;

        while (curr!=null){
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
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


    //225. Implement Stack using Queues
    class MyStack {
        private Queue<Integer> queue1;
        private Queue<Integer> queue2;

        public MyStack() {
            queue1 = new LinkedList<>();
            queue2 = new LinkedList<>();
        }

        public void push(int x) {
            // Enqueue new element to queue2
            queue2.add(x);

            // Dequeue all elements from queue1 and enqueue to queue2
            while (!queue1.isEmpty()) {
                queue2.add(queue1.remove());
            }

            // Swap the names of the two queues
            Queue<Integer> temp = queue1;
            queue1 = queue2;
            queue2 = temp;
        }

        public int pop() {
            if (queue1.isEmpty()) {
                throw new RuntimeException("Stack is empty");
            }
            return queue1.remove();
        }

        public int top() {
            if (queue1.isEmpty()) {
                throw new RuntimeException("Stack is empty");
            }
            return queue1.peek();
        }

        public boolean empty() {
            return queue1.isEmpty();
        }
    }

    //226. Invert Binary Tree
    public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode node = new TreeNode(root.val);
        node.right = invertTree(root.left);
        node.left = invertTree(root.right);
        return node;
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

    //283. Move Zeroes
    public void moveZeroes(int[] nums) {
        int count = 0;  // position to place the next non-zero value

        // First loop: Move all non-zeroes to the beginning
        for (int num : nums) {
            if (num != 0) {
                nums[count++] = num;
            }
        }

        // Fill in zeroes for the remaining positions
        while (count < nums.length) {
            nums[count++] = 0;
        }
    }


    //287. Find the Duplicate Number
        public int findDuplicate(int[] nums) {
            int slow = 0, fast = 0;
            while (true) {
                slow = nums[slow];// same as slow.next()
                fast = nums[nums[fast]];// same as fast.next().next()
                if (slow == fast) {
                    break;
                }
            }

            int slow2 = 0;
            while (true) {
                slow = nums[slow];
                slow2 = nums[slow2];
                if (slow == slow2) {
                    return slow;
                }
            }
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

    //403. Frog Jump
    public boolean canCross(int[] stones) {
        if (stones == null || stones.length == 0) {
            return false;
        }

        int n = stones.length;
        if (n == 1) {
            return true;
        }

        if (stones[1] != 1) {
            return false;
        }

        // Check if the gap is too large
        for (int i = 1; i < n; i++) {
            if (stones[i] - stones[i-1] > i) {
                return false;
            }
        }

        // Last stone's position
        int lastStone = stones[n - 1];

        // Set of stones for O(1) lookup
        Set<Integer> stonePositions = new HashSet<>();
        for (int stone : stones) {
            stonePositions.add(stone);
        }

        // For memoization
        Set<String> visited = new HashSet<>();

        return canCross(stonePositions, visited, 1, 1, lastStone);
    }

    private boolean canCross(Set<Integer> stonePositions, Set<String> visited, int position, int jump, int lastStone) {
        // Convert position and jump to a unique string key (for memoization)
        String key = position + "-" + jump;

        // If we've seen this combination before, return false
        if (visited.contains(key)) {
            return false;
        } else {
            visited.add(key);
        }

        // If we're at the last stone, return true
        if (position == lastStone) {
            return true;
        }

        // Check the possible jump sizes: k-1, k, k+1
        for (int i = -1; i <= 1; i++) {
            int newJump = jump + i;
            int newPosition = position + newJump;

            if (newJump > 0 && stonePositions.contains(newPosition)) {
                if (canCross(stonePositions, visited, newPosition, newJump, lastStone)) {
                    return true;
                }
            }
        }

        return false;
    }

    //424. Longest Repeating Character Replacement
    public int characterReplacement(String s, int k) {
        int size = s.length();
        int max = 0;
        int[] charCount = new int[26]; // To keep track of character counts within the window
        int maxCount = 0; // The character with the maximum count in the current window
        int left = 0;

        for (int right = 0; right < size; right++) {
            charCount[s.charAt(right) - 'A']++; // Increment the count of the current character
            maxCount = Math.max(maxCount, charCount[s.charAt(right) - 'A']);

            // If the difference between the current window size and the max character count
            // is greater than k, then we need to shrink the window from the left.
            if (right - left + 1 - maxCount > k) {
                charCount[s.charAt(left) - 'A']--; // Decrease the count of the character going out of the window
                left++;
            }

            max = Math.max(max, right - left + 1);
        }

        return max;
    }

    //543. Diameter of Binary Tree
    int result = -1;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return result;
    }

    private int dfs(TreeNode current) {
        if (current == null) {
            return -1;
        }
        int left = 1 + dfs(current.left);
        int right = 1 + dfs(current.right);
        result = Math.max(result, (left + right));
        return Math.max(left, right);
    }
    //567. Permutation in String
    public boolean checkInclusion(String s1, String s2) {
        int sizeOne = s1.length();
        int sizeTwo = s2.length();

        //Create two arrays to store occurrences for all alphabets
        int[] arrOne = new int[26];
        int[] arrTwo = new int[26];

        if (sizeOne > sizeTwo){
            return false;
        }

        for (int i = 0; i<sizeOne; i++){
            arrOne[s1.charAt(i)-'a']++;
            arrTwo[s2.charAt(i)-'a']++;
        }

        if (Arrays.equals(arrOne,arrTwo)){
            return true;
        }

        int front = 0;
        int back = sizeOne;

        while (back < sizeTwo){
            arrOne[s1.charAt(front)-'a']--;
            arrTwo[s2.charAt(back)-'a']++;

            if (Arrays.equals(arrOne,arrTwo)){
                return true;
            }

            front++;
            back++;
        }

        return false;
    }

    //646. Maximum Length of Pair Chain
    public int findLongestChain(int[][] pairs) {
        // Sort pairs based on their end times
        Arrays.sort(pairs, Comparator.comparingInt(a -> a[1]));

        // Initialize
        int current = Integer.MIN_VALUE;
        int count = 0;

        // Iterate through the sorted pairs
        for (int[] pair : pairs) {
            if (pair[0] > current) {
                count++;
                current = pair[1];
            }
        }

        return count;
    }

    //704. Binary Search
    public int search2(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;

        while(left <= right) {
            int mid = left + (right - left)/2;
            int midValue = nums[mid];

            if(midValue == target){
                return mid;
            } else if (midValue <= target){
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }

    //739. Daily Temperatures
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int idx = stack.pop();
                res[idx] = i - idx;
            }
            stack.push(i);
        }
        return res;
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

    //875. Koko Eating Bananas
    public int minEatingSpeed(int[] piles, int h) {
        int max = Arrays.stream(piles).max().getAsInt();  // Find the max value of piles
        int pileSize = piles.length;

        int left = 1;
        int right = max;

        while (left <= right) {
            int mid = (left + right) / 2;
            long count = 0;

            for (int pile : piles) {
                count += (int) Math.ceil((double) pile / mid);
            }

            if (count > h) { // If it takes more than h hours at this speed, increase speed.
                left = mid + 1;
            } else { // If it takes less than or equal to h hours, try reducing speed.
                right = mid - 1;
            }
        }

        return left; // By the end of the loop, left will be the smallest eating speed for which Koko can eat all bananas in h hours.

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

    //1436. Destination City
    public String destCity(List<List<String>> paths) {
        HashMap<String, String> hashMap = new HashMap<>();

        for (List<String> path : paths){
            hashMap.put(path.get(0),path.get(1));
        }

        for (String city :hashMap.values()){
            if (!hashMap.containsKey(city)){
                return city;
            }
        }
        return null;
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

    //2483. Minimum Penalty for a Shop
    /*public int bestClosingTime(String customers) {
        int hours = customers.length();
        int closingTime = 0;
        int minPenalty = Integer.MAX_VALUE;

        for (int i = 0; i < hours; i++){
            String open = customers.substring(0,i);
            String closed = customers.substring(i);

            int penalty = 0;
            for (int j = 0; j < open.length(); j ++){
                char customer = open.charAt(j);
                if (customer == 'N'){
                    penalty++;
                }
            }

            for (int k = 0; k < closed.length(); k ++){
                char customer = closed.charAt(k);
                if (customer == 'Y'){
                    penalty++;
                }
            }
            if (penalty < minPenalty){
                closingTime = i;
                minPenalty = penalty;
            }
        }
        return closingTime;
    }*/
    public int bestClosingTime(String customers) {
        //Loops through the customers once and update the score
        //If c is 'Y':
        //This means a customer is visiting the shop at this hour, so we increment score by 1. This is because if the shop were closed at this hour, it would incur a penalty.
        //
        //If c is 'N':
        //No customer is visiting, so we decrement score by 1. This implies that keeping the shop open at this hour would be wasteful and result in a penalty.
        int max_score = 0;
        int score = 0;
        int best_hour = -1;
        for(int i = 0; i < customers.length(); ++i) {
            score += (customers.charAt(i) == 'Y') ? 1 : -1;
            if(score > max_score) {
                max_score = score;
                best_hour = i;
            }
        }
        return best_hour + 1;
    }
}
