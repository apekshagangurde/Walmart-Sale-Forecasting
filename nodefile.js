// let course = {
//     title: "hld",
//     description: "projects",
//     rating: 5,
// };

// console.log(course);
// console.log(typeof (course));

// console.log(course.title);
// console.log(course['description']);

// let x = "apeksha";
// let y = x;
// x = "Mansi";
// console.log(x);
// console.log(y);

// let p = { name: 'Apeksha' };
// let q = p;

// p.name = "mansi";
// console.log(p);
// console.log(q);

// let course = ['hid', 'lid', 'dsa', 6, true, null];

// console.log(course[0]);
// console.log(course[1]);
// console.log(course[3]);
// console.log(course)

//functtion,hoisting

// console.log(a);
// var a = 10;

// console.log(a);
// console.log(this.a);
// console.log(window.a);
// console.log(window  );
// console.log(this === window);
//const let -> block scope and var->function scope

// {
//     let a = 10;
//     const b = 20;
//     var c = 30;
//     console.log(a);
//     console.log(b);
//     console.log(c);

// }
// console.log(c);


// function hello() {
//     let x = 10;
//     console.log(x);
// }
// let x = 100;
// hello();

// function add(a, b) {
//     return a + b;
// }
// console.log(add);
// console.log(add(2, 3));

// let sum = function (a, b) {
//     return a + b;
// }

// console.log(sum);
// console.log(sum(2, 3));


let sum = function (a, b) {
    return a + b;

}

let diff = function (a, b) {
    return a - b;
}

function operate(operationfunc, a, b) {
    return operationfunc(a, b);
}

console.log(operate(sum, 2, 3));
console.log(operate(diff, 2, 3));