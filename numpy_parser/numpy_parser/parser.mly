%token VAR PARAM CST
%token LPAR RPAR LBRACE RBRACE EQUAL COMMA
%token BINOP MINUS
%token NUMPY_FUNC
%token EOF

%start function_expr
%type <unit> function_expr

%%

function_expr: expr EOF { () }

binop:
  | BINOP { () }
  | MINUS { () }

args:
  | expr { () }
  | PARAM EQUAL expr { () }

expr:
  | VAR { () }
  | CST { () }
  | LPAR expr RPAR { () }
  | MINUS expr { () }
  | expr binop expr { () }
  | VAR LBRACE CST RBRACE { () }
  | NUMPY_FUNC LPAR separated_list(COMMA, args) RPAR { () }