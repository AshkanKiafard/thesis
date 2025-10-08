%%
%% webisthesis.cls
%% ===============
%%
%% Options (\documentclass[<comma-separated options list>]{webisthesis})
%% ---------------------------------------------------------------------
%% - english      For a thesis written in English
%% - german       For a thesis written in German
%% - bachelor     For a bachelor thesis (B.Sc. Informatik)
%% - master       For a master thesis   (M.Sc. Computer Science for Digital Media)
%% - buw          For a thesis in Weimar 
%% - ul           For a thesis in Leipzig
%% - fsu          For a thesis in Jena
%% - pb           For a thesis in Paderborn
%% Default: [german,bachelor,buw]
%%
%% New commands
%% ------------
%% \ThesisSetTitle{<title>}
%% \ThesisSetKeywords{<comma-and-space-separated list of keywords>} (optional)
%% \ThesisSetAuthor{<your first name> <your last name>}
%% \ThesisSetStudentNumber{<your number>}
%% \ThesisSetDateOfBirth{<dd>}{<mm>}{<yyyy of your birthday>}
%% \ThesisSetPlaceOfBirth{<location you were born>}
%% \ThesisSetSupervisors{<comma-separated list of your supervisors>}
%% \ThesisSetSubmissionDate{<dd>}{<mm>}{<yyyy of your final submission>}
%% \ThesisSetLocation{<your current location>} (optional, default: Weimar)
%% \ThesisSetProgramme{<custom degree programme name>} (optional)
%%   These commands can be used to specify some of the values that are used for
%%   generating the title page, the declaration of independent work and PDF meta
%%   attributes of the document (when \usehyperref is also used).
%%   Besides the commands marked (optional), all of these have to be specified
%%   before \begin{document} or else compilation will abort.
%% 
%% \usehyperref[<additional options>]
%%   This new command allows you to load the hyperref package with some suggested
%%   options (also setting PDF fields (like author)). Because of compatibility
%%   issues, hyperref has to be loaded after most packages, but also before some
%%   certain packages. Use this command at the appropriate place.
%%   The optional parameter can be used to forward additional options to
%%   hyperref.
%%   For instance, \usehyperref[breaklinks=true] to allow linebreaks within links.
%%
%% \formatdateshort{<dd>}{<mm>}{<yyyy>}
%%   Analogous to \formatdate{<dd>}{<mm>}{<yyyy>} of the datetime package, but
%%   prints a shorter version to be used in tables or figures.
%%
%% New environments
%% ----------------
%% frontmatter
%%  This will print the title page and the declaration of independent work.
%%  The pagenumbers will be turned to arabic and reset when this environment
%%  ends.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Required packages:
%% - babel
%% - caption
%% - datetime
%% - fancyhdr
%% - fontenc
%% - geometry
%% - hyperref
%% - ifpdf
%% - ifthen
%% - inputenc
%% - tocbibind
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Current maintainer: Johannes Kiesel
%% Based on work of Peter Prettenhofer and
%% Stefan Brecheisen (Institut fuer Informatik der LMU Muenchen)
%%

\def\fileversion{v1.5}
\def\filedate{2021/09/08}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{webisthesis}[\filedate\space\fileversion]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load base class (report) and basic utilities
\LoadClass[a4paper,12pt]{report}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{ifpdf}
\RequirePackage{ifthen}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options

% Language
\newcommand{\documentlanguage}{}
\DeclareOption{english}{%
  \renewcommand{\documentlanguage}{english}%
}
\DeclareOption{german}{%
  \renewcommand{\documentlanguage}{ngerman}%
}
\ExecuteOptions{german} % default

% Type
\newcommand{\thesistype}{}
\DeclareOption{bachelor}{%
  \renewcommand{\thesistype}{Bachelor}%
}
\DeclareOption{master}{%
  \renewcommand{\thesistype}{Master}%
}
\ExecuteOptions{bachelor} % default

% University
\newcommand{\thesisuniversity}{}
\DeclareOption{buw}{%
    \renewcommand{\thesisuniversity}{buw}
}
\DeclareOption{ul}{%
    \renewcommand{\thesisuniversity}{ul}
}
\DeclareOption{fsu}{%
    \renewcommand{\thesisuniversity}{fsu}
}
\DeclareOption{pb}{%
    \renewcommand{\thesisuniversity}{pb}
}
\ExecuteOptions{buw} % default

% Web publishing
\newcommand{\thesisweb}{0}
\DeclareOption{web}{%
    \renewcommand{\thesisweb}{1}
}

\ProcessOptions%

% Language
\RequirePackage[\documentlanguage]{babel}
% Datetime
% Add ``.'' after abbreviated months
\expandafter\newcommand\csname shortdateperiod1\endcsname{.}
\expandafter\newcommand\csname shortdateperiod2\endcsname{.}
\expandafter\newcommand\csname shortdateperiod3\endcsname{.}
\expandafter\newcommand\csname shortdateperiod4\endcsname{.}
\expandafter\newcommand\csname shortdateperiod5\endcsname{}% May
\expandafter\newcommand\csname shortdateperiod6\endcsname{e}% June
\expandafter\newcommand\csname shortdateperiod7\endcsname{y}% July
\expandafter\newcommand\csname shortdateperiod8\endcsname{.}
\expandafter\newcommand\csname shortdateperiod9\endcsname{.}
\expandafter\newcommand\csname shortdateperiod10\endcsname{.}
\expandafter\newcommand\csname shortdateperiod11\endcsname{.}
\expandafter\newcommand\csname shortdateperiod12\endcsname{.}
\newcommand{\formatdateshort}[3]{{\shortdate\formatdate{#1}{#2}{#3}}}
\ifthenelse{\equal{\documentlanguage}{english}}{%
  \RequirePackage[us,nodayofweek]{datetime}%
  \renewcommand{\formatdateshort}[3]{\shortmonthname[#2]{}\csname shortdateperiod#2\endcsname{}\ #1, #3}%
}{}
\ifthenelse{\equal{\documentlanguage}{ngerman}}{%
  \RequirePackage[nodayofweek]{datetime}%
  \newdateformat{germanlongdate}{\THEDAY.\ \monthname[\THEMONTH]{} \THEYEAR}%
  \germanlongdate%
  %\renewcommand{\dateseparator}{.}%
  \renewcommand{\formatdateshort}[3]{\twodigit{#1}.\twodigit{#2}.#3}%
}{}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables
\newtoks\thesistitle
\newcommand{\ThesisSetTitle}[1]{\global\thesistitle{#1}}

\newtoks\thesisauthor
\newcommand{\ThesisSetAuthor}[1]{\global\thesisauthor{#1}}

\newtoks\thesissupervisors
\newcommand{\ThesisSetSupervisors}[1]{\global\thesissupervisors{#1}}

\newtoks\thesissubmissiondate
\newcommand{\ThesisSetSubmissionDate}[3]{\global\thesissubmissiondate{\formatdate{#1}{#2}{#3}}}

\newtoks\thesislocation
\newcommand{\ThesisSetLocation}[1]{\global\thesislocation{#1}}
\ThesisSetLocation{Weimar} % default

\newtoks\thesisprogramme
\newcommand{\ThesisSetProgramme}[1]{\global\thesisprogramme{#1}}

\newtoks\thesisauthorstudentnumber
\newcommand{\ThesisSetStudentNumber}[1]{\global\thesisauthorstudentnumber{#1}}

\newtoks\thesisauthordateofbirth
\newcommand{\ThesisSetDateOfBirth}[3]{\global\thesisauthordateofbirth{\formatdateshort{#1}{#2}{#3}}}

\newtoks\thesisauthorplaceofbirth
\newcommand{\ThesisSetPlaceOfBirth}[1]{\global\thesisauthorplaceofbirth{#1}}

\newtoks\thesiskeywords
\newcommand{\ThesisSetKeywords}[1]{\global\thesiskeywords{#1}}

% Check if variables are set
% \checkthesisvariable{variable value}{name}{setter command}{example}
\newcommand{\checkthesisvariable}[4]{%
  \ifthenelse{\equal{#1}{}}{%
    \ClassWarning{webisthesis}{%
      #2 was not defined! %
      Use command "#3{#2}" (e.g.: "#4") before "begin{document}"}%
    \global\thesisvariableerrors{true}%
  }{}%
}
\newcommand{\checkthesisvariabledate}[3]{%
  \ifthenelse{\equal{#1}{}}{%
    \ClassWarning{webisthesis}{%
      #2 was not defined! %
      Use command "#3{day}{month}{year}" (e.g.: "#3{31}{12}{2022}") before "begin{document}"}%
    \global\thesisvariableerrors{true}%
  }{}%
}
% Checking all variables...
\newcommand{\checkthesisvariables}{%
  \newtoks\thesisvariableerrors%
  \checkthesisvariable{\the\thesistitle}{Title}{ThesisSetTitle}{Title of my Thesis}%
  \checkthesisvariable{\the\thesisauthor}{Author}{ThesisSetAuthor}{First name Last name}%
  \checkthesisvariable{\the\thesissupervisors}{Supervisors}{ThesisSetSupervisors}{Prof. Dr. Benno Stein,Prof. Albert Einstein}%
  \checkthesisvariabledate{\the\thesissubmissiondate}{Submission date}{ThesisSetSubmissionDate}%
  \checkthesisvariable{\the\thesisauthorstudentnumber}{Student Number}{ThesisSetStudentNumber}{01234}%
  \checkthesisvariabledate{\the\thesisauthordateofbirth}{Date of Birth (Author)}{ThesisSetDateOfBirth}%
  \checkthesisvariable{\the\thesisauthorplaceofbirth}{Place of Birth (Author)}{ThesisSetPlaceOfBirth}{Duckburg}%
  \ifthenelse{\equal{\the\thesisvariableerrors}{true}}{%
    \ClassError{webisthesis}{%
      Not all required variables for the thesis template have been set}{%
      See the Warnings for class webisthesis above.}%
  }{}%
}
% ... at the begin of the document
\AtBeginDocument{\checkthesisvariables}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% University-specific template words
\newtoks\thesisfrontpageuniversity
\newtoks\thesisfrontpagefaculty
\newtoks\thesisfrontpagecourse
\newtoks\thesisfrontpagetypename

\ifthenelse{\equal{\documentlanguage}{english}}{%
  \ifthenelse{\equal{\thesisuniversity}{buw}}{%
    \global\thesisfrontpageuniversity{Bauhaus-Universit{{\"a}}t Weimar}
    \global\thesisfrontpagefaculty{Faculty of Media}
    \global\thesisfrontpagecourse{Degree Programme }
  }{}
  \ifthenelse{\equal{\thesisuniversity}{ul}}{%
    \global\thesisfrontpageuniversity{Leipzig University} 
    \global\thesisfrontpagefaculty{Institute of Computer Science}
    \global\thesisfrontpagecourse{Degree Programme }
  }{}
  \ifthenelse{\equal{\thesisuniversity}{fsu}}{% 
    \global\thesisfrontpageuniversity{Friedrich-Schiller-Universität Jena} 
    \global\thesisfrontpagefaculty{Institute of Computer Science}
    \global\thesisfrontpagecourse{Degree Programme }
  }{}
  \ifthenelse{\equal{\thesisuniversity}{pb}}{% 
    \global\thesisfrontpageuniversity{Paderborn University} 
    \global\thesisfrontpagefaculty{Institute of Computer Science}
    \global\thesisfrontpagecourse{Degree Programme }
  }{}
}{}
\ifthenelse{\equal{\documentlanguage}{ngerman}}{%
  \ifthenelse{\equal{\thesisuniversity}{buw}}{%
    \global\thesisfrontpageuniversity{Bauhaus-Universit{\"a}t Weimar}
    \global\thesisfrontpagefaculty{Fakult{\"a}t Medien}
    \global\thesisfrontpagecourse{Studiengang }
  }{}
  \ifthenelse{\equal{\thesisuniversity}{ul}}{%
    \global\thesisfrontpageuniversity{Universit{\"a}t Leipzig} 
    \global\thesisfrontpagefaculty{Institut f{\"u}r Informatik}
    \global\thesisfrontpagecourse{Studiengang }
  }{}
  \ifthenelse{\equal{\thesisuniversity}{fsu}}{% 
    \global\thesisfrontpageuniversity{Friedrich-Schiller-Universität Jena} 
    \global\thesisfrontpagefaculty{Institut f{\"u}r Informatik}
    \global\thesisfrontpagecourse{Studiengang }
  }{}
  \ifthenelse{\equal{\thesisuniversity}{pb}}{% 
    \global\thesisfrontpageuniversity{Universit{\"a}t Paderborn} 
    \global\thesisfrontpagefaculty{Institut f{\"u}r Informatik}
    \global\thesisfrontpagecourse{Studiengang }
  }{} 
}{}

\ifthenelse{\equal{\thesistype}{Bachelor}}{%
  \ifthenelse{\equal{\documentlanguage}{ngerman}}{%
    \ifthenelse{\equal{\thesisuniversity}{buw}}{%
      \global\thesisprogramme{Medieninformatik}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{ul}}{% 
      \global\thesisprogramme{Informatik, B.Sc.}
      %\global\thesisprogramme{Digital Humanities, B.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{fsu}}{% 
      \global\thesisprogramme{Informatik, B.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{pb}}{% 
      \global\thesisprogramme{Informatik, B.Sc.}
    }{}
  }{}
  \ifthenelse{\equal{\documentlanguage}{english}}{%
    \ifthenelse{\equal{\thesisuniversity}{buw}}{%
      \global\thesisprogramme{Medieninformatik}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{ul}}{% 
      \global\thesisprogramme{Computer Science, B.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{fsu}}{% 
      \global\thesisprogramme{Computer Science, B.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{pb}}{% 
      \global\thesisprogramme{Computer Science, B.Sc.}
    }{}
  }{}
}{}
\ifthenelse{\equal{\thesistype}{Master}}{%
  \ifthenelse{\equal{\documentlanguage}{ngerman}}{%
    \ifthenelse{\equal{\thesisuniversity}{buw}}{%
      \global\thesisprogramme{Computer Science for Digital Media}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{ul}}{% 
      \global\thesisprogramme{Informatik, M.Sc.}
      %\global\thesisprogramme{Digital Humanities, M.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{fsu}}{% 
      \global\thesisprogramme{Informatik, M.Sc.}
      %\global\thesisprogramme{Computational and Data Science, M.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{pb}}{% 
      \global\thesisprogramme{Informatik, M.Sc.}
    }{}
  }{}
    \ifthenelse{\equal{\documentlanguage}{english}}{%
    \ifthenelse{\equal{\thesisuniversity}{buw}}{%
      \global\thesisprogramme{Computer Science for Digital Media}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{ul}}{%  
      \global\thesisprogramme{Computer Science, M.Sc.}
      %\global\thesisprogramme{Digital Humanities, M.Sc.}
      %\global\thesisprogramme{Data Science, M.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{fsu}}{% 
      \global\thesisprogramme{Computer Science, M.Sc.}
      %\global\thesisprogramme{Computational and Data Science, M.Sc.}
    }{}
    \ifthenelse{\equal{\thesisuniversity}{pb}}{% 
      \global\thesisprogramme{Computer Science, M.Sc.}
    }{}
  }{}
}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% University-agnostic words
\newtoks\thesisfrontpagestudentnumber
\newtoks\thesisfrontpageborn
\newtoks\thesisfrontpagesupervisor
\newtoks\thesisfrontpagedateofsubmission
\newtoks\thesisdeclarationtitle
\newtoks\thesisdeclarationtext
\newtoks\thesisfrontmatterkeywords
\ifthenelse{\equal{\documentlanguage}{english}}{%
  \global\thesisfrontpagetypename{'s Thesis}
  \global\thesisfrontpagestudentnumber{Matriculation Number}
  \global\thesisfrontpageborn{Born \the\thesisauthordateofbirth{} in %
    \the\thesisauthorplaceofbirth}
  \global\thesisfrontpagesupervisor{Referee}
  \global\thesisfrontpagedateofsubmission{Submission date}
  \global\thesisdeclarationtitle{Declaration}
  \global\thesisdeclarationtext{Unless otherwise indicated in the text or %
    references, this thesis is entirely the product of my own scholarly work.}
  \global\thesisfrontmatterkeywords{Keywords}
}{}
\ifthenelse{\equal{\documentlanguage}{ngerman}}{%
  \global\thesisfrontpagetypename{arbeit}
  \global\thesisfrontpagestudentnumber{Matrikelnummer}
  \global\thesisfrontpageborn{geb.\ am: \the\thesisauthordateofbirth{} in %
    \the\thesisauthorplaceofbirth}
  \global\thesisfrontpagesupervisor{Gutachter}
  \global\thesisfrontpagedateofsubmission{Datum der Abgabe}
  \global\thesisdeclarationtitle{Erkl{\"a}rung}
  \global\thesisdeclarationtext{Hiermit versichere ich, dass ich diese Arbeit %
    selbstst{\"a}ndig verfasst und keine anderen als die angegebenen Quellen und %
    Hilfsmittel verwendet habe.}
  \global\thesisfrontmatterkeywords{Schl{\"u}sselworte}
}{}

% Frontpage
\newcommand{\frontpage}{%
  \begin{titlepage}
		\vspace{2cm}
		\hspace*{-1cm}
    \parbox{10cm}{\normalsize
      \the\thesisfrontpageuniversity\\
      \the\thesisfrontpagefaculty\\
      \the\thesisfrontpagecourse\the\thesisprogramme}

		\begin{center}
			\vspace{3cm}\Huge
      {\bf\the\thesistitle}

      \vspace{3cm}\Huge
      {\bf\thesistype\the\thesisfrontpagetypename}
		\end{center}
    \vspace{\fill}\normalsize
    \the\thesisauthor\hfill\if\thesisweb0 \the\thesisfrontpagestudentnumber{} \the\thesisauthorstudentnumber\fi\\%
    \if\thesisweb0 \the\thesisfrontpageborn\fi\\[3em]
    \newcounter{supervisors}\makeatletter%
    \@for \name:=\the\thesissupervisors\do{%
      \addtocounter{supervisors}{1}\arabic{supervisors}. %
      \the\thesisfrontpagesupervisor: \name \\%
    }%
    \makeatother\\[3em]%
    \the\thesisfrontpagedateofsubmission: \the\thesissubmissiondate
  \end{titlepage}
}%

% Declaration of independent work
\newcommand{\declaration}{
  \begin{titlepage}
    \vspace*{\fill}
    \parindent 0cm
    \begin{center}
      \textbf{\huge \the\thesisdeclarationtitle}
    \end{center}

    \vspace{1cm}

		\large \the\thesisdeclarationtext

    \vspace{1cm}
    \the\thesislocation, \the\thesissubmissiondate

    \vspace{2cm}
    \makebox[10cm]{\dotfill}\\
    \the\thesisauthor

    \vspace*{\fill}
  \end{titlepage}
}

\newenvironment{frontmatter}{%
  \pagenumbering{roman}%
  \frontpage%
  \declaration%
}{%
  \clearpage%
  \pagenumbering{arabic}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Page layout
\RequirePackage{geometry}
\RequirePackage{fancyhdr}

\geometry{body={140mm,210mm},footskip=12mm}
\pagestyle{fancy}
\headheight 14pt
\fancyhf{}
\fancyhead[L]{\small\slshape\leftmark}
\fancyfoot[C]{\thepage}

% Enumerate subsections and include them in the table of contents
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}

% List bibliography in table of contents
\RequirePackage[nottoc,notlot,notlof]{tocbibind}

% Specification of figure/table caption style
\RequirePackage[font=small,labelfont=bf]{caption}

% This new command allows you to load the hyperref package with some suggested
% options (also setting PDF fields (like author)). Because of compatibility
% issues, hyperref has to be loaded after most packages, but also before some
% certain packages. Use this command at the appropriate place.
%
% The optional parameter can be used to forward additional options to hyperref.
% e.g. \usehyperref[breaklinks=true] to allow linebreaks within links.
\newcommand{\usehyperref}[1][]{%
  \newcommand{\hyperoptions}{%
    colorlinks=false,%
    pageanchor=false%
  }%
  \ifpdf%
    \newcommand{\hyperpdfoptions}{%
      ,pdftex,%
      pdfborder={0 0 0},%
      pdfauthor={\the\thesisauthor},%
      pdftitle={\the\thesistitle},%
      pdfkeywords={\the\thesiskeywords},%
      pdfdisplaydoctitle,%
      pdfpagemode={UseNone},%
      pdfstartview={Fit}%
    }%
    \usepackage[\hyperoptions\hyperpdfoptions,#1]{hyperref}
  \else%
    \usepackage[\hyperoptions,#1]{hyperref}
  \fi%
}

