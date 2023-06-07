import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

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
}
