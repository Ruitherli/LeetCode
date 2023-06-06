import java.util.Arrays;

public class Solution {

     //Definition for singly-linked list.
     public class ListNode {
         int val;
         ListNode next;
         ListNode() {}
         ListNode(int val) { this.val = val; }
         ListNode(int val, ListNode next) { this.val = val; this.next = next; }
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
}
