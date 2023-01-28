{
  open Parser

  let check_numpy_func =
    let numpy_hash = Hashtbl.create 10 in

    List.iter (fun (math_fun) -> Hashtbl.add numpy_hash math_fun true)
    [
      "np.sin";
      "np.cos";
      "np.tan";
      "np.arcsin";
      "np.arccos";
      "np.arctan";
      "np.sinh";
      "np.cosh";
      "np.tanh";
      "np.arcsinh";
      "np.arccosh";
      "np.arctanh";
      "np.exp";
      "np.log";
      "np.log2";
      "np.log10";
      "np.sqrt";
      "np.power";
      "np.abs"
    ];

    fun func_name -> Hashtbl.mem numpy_hash func_name

  let fail s =
    failwith (Printf.sprintf "Unexpected token: %s" s)

}

let digit  = ['0'-'9']
let number = digit+
let alpha  = ['a'-'z' 'A'-'Z']
let space  = [' ' '\t' '\r']

let float   = number ('.' number)?
let cst     = float | "np.pi" | "np.e" | "10e" '-'? number
let var     = "x"
let np_func = "np." alpha+
let param   = "axis" | "ord" | "p"


rule token = parse
  | space+             { token lexbuf }
  | "+"                { BINOP }
  | "-"                { MINUS }
  | "*"                { BINOP }
  | "/"                { BINOP }
  | "**"               { BINOP }
  | ","                { COMMA }
  | "("                { LPAR }
  | "["                { LBRACE }
  | ")"                { RPAR }
  | "]"                { RBRACE }
  | "="                { EQUAL }
  | cst                { CST }
  | var                { VAR }
  | np_func as np_func { if check_numpy_func np_func then NUMPY_FUNC else fail np_func }
  | param              { PARAM }
  | eof                { EOF }
  | _ as c             { fail String.(make 1 c)}